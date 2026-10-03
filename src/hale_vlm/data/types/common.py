"""Shared modality enums."""

from enum import StrEnum


class Modality(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"
    MULTI_IMAGE = "multi_image"
