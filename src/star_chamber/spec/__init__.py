"""Council protocol specification files shipped as package data, and the version they state."""

SPEC_VERSION = "0.11"
"""The protocol version this package implements, as SPEC.md states it.

A consumer compares this with the version its own instructions were written
against. It advances when the protocol changes, which the package version does
not track, because a patch release can leave the protocol untouched.
"""
