import sys
import hashlib


def hash_input(val):
    hash_object = hashlib.sha1(val.encode("utf-8"))
    return hash_object.digest()
