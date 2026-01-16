#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "bargain_game.h"
#include "bargain_kernels.cuh"

namespace py = pybind11;

class BargainGameEnv {
private:
    GameBatch* batch_;
    uint32_t num_games_;
    int device_id_;
    bool self_play_;
    uint64_t seed_;
    bool rng_initialized_;

    // Device tensors for actions
    torch::Tensor d_actions_;

public:
    BargainGameEnv(uint32_t num_games, bool self_play = true, int device = 0)
        : num_games_(num_games), device_id_(device), self_play_(self_play),
          seed_(0), rng_initialized_(false) {

        // Create game batch
        batch_ = create_game_batch(num_games, device);

        // Allocate action tensor on device
        auto options = torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(torch::kCUDA, device_id_);
        d_actions_ = torch::zeros({(int64_t)num_games_}, options);
    }

    ~BargainGameEnv() {
        if (batch_) {
            destroy_game_batch(batch_);
            batch_ = nullptr;
        }
    }

    // Prevent copying
    BargainGameEnv(const BargainGameEnv&) = delete;
    BargainGameEnv& operator=(const BargainGameEnv&) = delete;

    uint32_t num_games() const { return num_games_; }
    int device() const { return device_id_; }
    bool self_play() const { return self_play_; }

    std::tuple<torch::Tensor, torch::Tensor> reset(uint64_t seed = 0) {
        // Always reinitialize RNG on reset for determinism
        // (calling reset with same seed should give same results)
        seed_ = seed;
        init_rng(batch_, seed_);
        rng_initialized_ = true;

        // Reset all games
        reset_games(batch_);
        cudaDeviceSynchronize();

        // Create output tensors that wrap device memory
        auto float_options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCUDA, device_id_);

        // Copy observations to new tensor
        torch::Tensor observations = torch::from_blob(
            batch_->d_observations,
            {(int64_t)num_games_, OBS_TOTAL_DIM},
            float_options
        ).clone();

        // Copy action masks to new tensor
        torch::Tensor action_masks = torch::from_blob(
            batch_->d_action_masks,
            {(int64_t)num_games_, NUM_TOTAL_ACTIONS},
            float_options
        ).clone();

        return std::make_tuple(observations, action_masks);
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    step(torch::Tensor actions) {
        // Validate input
        TORCH_CHECK(actions.dim() == 1, "Actions must be 1D tensor");
        TORCH_CHECK(actions.size(0) == (int64_t)num_games_, "Actions size must match num_games");
        TORCH_CHECK(actions.device().is_cuda(), "Actions must be on CUDA device");
        TORCH_CHECK(actions.dtype() == torch::kInt32, "Actions must be int32");

        // Step all games
        step_games(batch_, actions.data_ptr<int32_t>());
        cudaDeviceSynchronize();

        // Create output tensors
        auto float_options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCUDA, device_id_);
        auto uint8_options = torch::TensorOptions()
            .dtype(torch::kUInt8)
            .device(torch::kCUDA, device_id_);

        torch::Tensor observations = torch::from_blob(
            batch_->d_observations,
            {(int64_t)num_games_, OBS_TOTAL_DIM},
            float_options
        ).clone();

        torch::Tensor action_masks = torch::from_blob(
            batch_->d_action_masks,
            {(int64_t)num_games_, NUM_TOTAL_ACTIONS},
            float_options
        ).clone();

        torch::Tensor rewards = torch::from_blob(
            batch_->d_rewards,
            {(int64_t)num_games_, 2},
            float_options
        ).clone();

        torch::Tensor dones = torch::from_blob(
            batch_->d_dones,
            {(int64_t)num_games_},
            uint8_options
        ).clone().to(torch::kBool);

        torch::Tensor truncateds = torch::from_blob(
            batch_->d_truncateds,
            {(int64_t)num_games_},
            uint8_options
        ).clone().to(torch::kBool);

        return std::make_tuple(observations, action_masks, rewards, dones, truncateds);
    }

    torch::Tensor get_current_player() {
        auto uint8_options = torch::TensorOptions()
            .dtype(torch::kUInt8)
            .device(torch::kCUDA, device_id_);

        return torch::from_blob(
            batch_->d_current_players,
            {(int64_t)num_games_},
            uint8_options
        ).clone();
    }

    torch::Tensor get_random_actions(uint8_t target_player) {
        auto int_options = torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(torch::kCUDA, device_id_);

        torch::Tensor actions = torch::zeros({(int64_t)num_games_}, int_options);
        generate_random_actions(batch_, actions.data_ptr<int32_t>(), target_player);
        cudaDeviceSynchronize();

        return actions;
    }

    void auto_reset() {
        auto_reset_done_games(batch_);
        cudaDeviceSynchronize();
    }

    torch::Tensor get_dones() {
        auto uint8_options = torch::TensorOptions()
            .dtype(torch::kUInt8)
            .device(torch::kCUDA, device_id_);

        return torch::from_blob(
            batch_->d_dones,
            {(int64_t)num_games_},
            uint8_options
        ).clone().to(torch::kBool);
    }
};

PYBIND11_MODULE(cuda_bargain_core, m) {
    m.doc() = "CUDA Bargaining Game Environment for Deep RL";

    py::class_<BargainGameEnv>(m, "BargainGameEnv")
        .def(py::init<uint32_t, bool, int>(),
             py::arg("num_games"),
             py::arg("self_play") = true,
             py::arg("device") = 0,
             "Create a batch of bargaining game environments on GPU")
        .def("reset", &BargainGameEnv::reset,
             py::arg("seed") = 0,
             "Reset all games. Returns (observations, action_masks)")
        .def("step", &BargainGameEnv::step,
             py::arg("actions"),
             "Step all games. Returns (obs, masks, rewards, dones, truncateds)")
        .def("get_current_player", &BargainGameEnv::get_current_player,
             "Get current player for each game (0 or 1)")
        .def("get_random_actions", &BargainGameEnv::get_random_actions,
             py::arg("target_player"),
             "Get random valid actions for target player")
        .def("auto_reset", &BargainGameEnv::auto_reset,
             "Auto-reset completed games")
        .def("get_dones", &BargainGameEnv::get_dones,
             "Get done flags for all games")
        .def_property_readonly("num_games", &BargainGameEnv::num_games)
        .def_property_readonly("device", &BargainGameEnv::device)
        .def_property_readonly("self_play", &BargainGameEnv::self_play);

    // Export constants
    m.attr("NUM_ACTIONS") = py::int_(NUM_TOTAL_ACTIONS);
    m.attr("OBS_DIM") = py::int_(OBS_TOTAL_DIM);
    m.attr("ACTION_ACCEPT") = py::int_(ACTION_ACCEPT);
    m.attr("ACTION_WALK") = py::int_(ACTION_WALK);
    m.attr("NUM_ITEM_TYPES") = py::int_(NUM_ITEM_TYPES);
    m.attr("MAX_ROUNDS") = py::int_(MAX_ROUNDS);
    m.attr("ITEM_QUANTITIES") = py::make_tuple(ITEM_QTY_0, ITEM_QTY_1, ITEM_QTY_2);
}
