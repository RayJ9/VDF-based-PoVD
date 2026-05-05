from __future__ import annotations

import hashlib
import math

from . import settings as platform_settings


def hash_data(*args: object) -> str:
    hasher = hashlib.sha256()
    for arg in args:
        hasher.update(str(arg).encode("utf-8"))
    return hasher.hexdigest()


def hash_to_int(data_hex: str) -> int:
    return int(data_hex, 16)


def g_func(
    hash_val: str,
    min_delay_rounds: int = platform_settings.MIN_DELAY_ROUNDS,
    max_delay_rounds: int = platform_settings.MAX_DELAY_ROUNDS,
) -> int:
    value = hash_to_int(hash_val)
    span = max_delay_rounds - min_delay_rounds + 1
    return min_delay_rounds + (value % span)


def map_to_group(hash_val: str, modulus: int) -> int:
    return hash_to_int(hash_val) % modulus


def get_prime(seed: str, coprime_to: int | None = None) -> int:
    candidate = (hash_to_int(seed) % 100000) + 3
    if candidate % 2 == 0:
        candidate += 1

    while True:
        is_prime = True
        upper_bound = int(candidate**0.5) + 1
        for value in range(3, upper_bound, 2):
            if candidate % value == 0:
                is_prime = False
                break
        if is_prime and (coprime_to is None or math.gcd(candidate, coprime_to) == 1):
            return candidate
        candidate += 2


class WesolowskiVDF:
    @staticmethod
    def setup(time_bound: int | None = None) -> int:
        return platform_settings.VDF_MODULUS_N

    @staticmethod
    def eval(modulus: int, z_value: int, squarings: int) -> tuple[int, int]:
        exponent = pow(2, squarings)
        y_value = pow(z_value, exponent, modulus)
        challenge = get_prime(hash_data(y_value, z_value, squarings), coprime_to=platform_settings.VDF_PHI)
        quotient = exponent // challenge
        proof = pow(z_value, quotient, modulus)
        return y_value, proof

    @staticmethod
    def verify(modulus: int, z_value: int, y_value: int, squarings: int, proof: int) -> bool:
        if z_value < 0 or z_value >= modulus or y_value < 0 or y_value >= modulus:
            return False
        if proof < 0 or proof >= modulus:
            return False

        challenge = get_prime(hash_data(y_value, z_value, squarings), coprime_to=platform_settings.VDF_PHI)
        epsilon = pow(2, squarings, challenge)
        return (pow(proof, challenge, modulus) * pow(z_value, epsilon, modulus)) % modulus == y_value
