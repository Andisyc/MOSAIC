# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class SuperviseLearning(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_student_obs,
        num_actions,
        student_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=0.1,
        **kwargs,
    ):
        if kwargs:
            print(
                "StudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)
        self.loaded_teacher = False  # indicates if teacher has been loaded

        mlp_input_dim_s = num_student_obs

        # ========== student ==========
        student_layers = []
        student_layers.append(nn.Linear(mlp_input_dim_s, student_hidden_dims[0]))
        student_layers.append(activation)
        for layer_index in range(len(student_hidden_dims)):
            if layer_index == len(student_hidden_dims) - 1:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], num_actions))
            else:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], student_hidden_dims[layer_index + 1]))
                student_layers.append(activation)
        self.student = nn.Sequential(*student_layers)

        print(f"Student MLP: {self.student}")

        # ========== action noise ==========
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None

        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None, hidden_states=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.student(observations)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations):
        self.update_distribution(observations)
        self.distribution.sample()
        return self.act_inference(observations)

    def act_inference(self, observations):
        actions_mean = self.student(observations)
        return actions_mean

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the student network.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.
        """

        # Load specifically for FrontRES / supervised learning
        if any("actor" in key for key in state_dict.keys()):  # adapting RL checkpoint format
            student_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    student_state_dict[key.replace("actor.", "")] = value
            self.student.load_state_dict(student_state_dict, strict=strict)
        else:
            super().load_state_dict(state_dict, strict=strict)
        return True

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass
