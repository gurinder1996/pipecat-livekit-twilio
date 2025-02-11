#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Custom exceptions for the conversation flow system.

This module defines the exception hierarchy used throughout the flow system:
- FlowError: Base exception for all flow-related errors
- FlowInitializationError: Initialization failures
- FlowTransitionError: State transition issues
- InvalidFunctionError: Function registration/calling problems
- ActionError: Action execution failures

These exceptions provide specific error types for better error handling
and debugging.
"""


class FlowError(Exception):
    """Base exception for all flow-related errors."""

    pass


class FlowInitializationError(FlowError):
    """Raised when flow initialization fails."""

    pass


class FlowTransitionError(FlowError):
    """Raised when a state transition fails."""

    pass


class InvalidFunctionError(FlowError):
    """Raised when an invalid or unavailable function is called."""

    pass


class ActionError(FlowError):
    """Raised when an action execution fails."""

    pass
