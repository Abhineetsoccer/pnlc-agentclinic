from enum import Enum


class TrajectoryField(str, Enum):
    SCENARIO_INDEX = "scenario_index"
    TURN_INDEX = "turn_index"
    STATE = "state"
    INCOMING_MESSAGE = "incoming_message"
    THOUGHT = "thought"
    DOCTOR_ACTION = "doctor_action"
    RAW_MODEL_OUTPUT = "raw_model_output"
    PARSED_OK = "parsed_ok"
