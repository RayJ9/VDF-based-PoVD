from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import time
from typing import Dict, Iterator, Optional

from . import settings as platform_settings

from .crypto import WesolowskiVDF, g_func, get_prime, hash_data, hash_to_int, map_to_group


@dataclass(frozen=True)
class PlatformConfig:
    num_miners: int = platform_settings.NUM_MINERS
    target_block_height: int = platform_settings.TARGET_BLOCK_HEIGHT
    propagation_vector: tuple[float, ...] = tuple(platform_settings.PROPAGATION_VECTOR_WD)
    vdf_modulus_n: int = platform_settings.VDF_MODULUS_N
    vdf_phi: int = platform_settings.VDF_PHI
    vdf_squarings_per_round: int = platform_settings.VDF_SQUARINGS_PER_ROUND
    pow_hashes_per_round: int = platform_settings.POW_HASHES_PER_ROUND
    min_delay_rounds: int = platform_settings.MIN_DELAY_ROUNDS
    max_delay_rounds: int = platform_settings.MAX_DELAY_ROUNDS
    pow_difficulty_phi: int = platform_settings.POW_DIFFICULTY_PHI
    max_round_multiplier: int = 500

    def __post_init__(self) -> None:
        required_fields = {
            "num_miners": self.num_miners,
            "target_block_height": self.target_block_height,
            "vdf_modulus_n": self.vdf_modulus_n,
            "vdf_phi": self.vdf_phi,
            "vdf_squarings_per_round": self.vdf_squarings_per_round,
            "pow_hashes_per_round": self.pow_hashes_per_round,
            "min_delay_rounds": self.min_delay_rounds,
            "max_delay_rounds": self.max_delay_rounds,
            "pow_difficulty_phi": self.pow_difficulty_phi,
        }
        missing_fields = [name for name, value in required_fields.items() if value is None]
        if missing_fields:
            raise ValueError(f"Missing platform settings: {', '.join(missing_fields)}")
        if self.num_miners < 2:
            raise ValueError("num_miners must be at least 2")
        if not self.propagation_vector:
            raise ValueError("propagation_vector cannot be empty")
        if self.min_delay_rounds > self.max_delay_rounds:
            raise ValueError("min_delay_rounds cannot exceed max_delay_rounds")
        if self.vdf_squarings_per_round <= 0 or self.pow_hashes_per_round <= 0:
            raise ValueError("workload parameters must be positive")
        if self.target_block_height <= 0:
            raise ValueError("target_block_height must be positive")


def build_platform_config(**overrides: object) -> PlatformConfig:
    return PlatformConfig(**overrides)


@dataclass
class MiningTick:
    block: Optional["Block"]
    work_units: int
    active_jobs: int = 1


@dataclass
class MinerStepReport:
    miner_id: int
    block: Optional["Block"]
    work_units: int
    active_jobs: int


@dataclass
class Block:
    height: int
    parent_hash: str
    proposer_id: int
    prev_vdf_y: int | str | None
    payload_hash: str
    vdf_y: Optional[int]
    vdf_proof: Optional[int]
    round_generated: int
    nonce: Optional[int] = None
    pow_hash: Optional[str] = None
    block_hash: str = field(init=False)

    def __post_init__(self) -> None:
        self.refresh_hash()

    def refresh_hash(self) -> None:
        self.block_hash = self.compute_hash()

    def compute_hash(self) -> str:
        return hash_data(
            self.height,
            self.parent_hash,
            self.proposer_id,
            self.prev_vdf_y,
            self.payload_hash,
            self.vdf_y,
            self.vdf_proof,
            self.round_generated,
            self.nonce,
            self.pow_hash,
        )

    def consensus_output(self) -> int | str:
        if self.pow_hash is not None:
            return self.pow_hash
        if self.vdf_y is not None:
            return self.vdf_y
        return hash_to_int(self.block_hash)

    def verify_parent_link(self, parent_block: "Block") -> bool:
        return str(self.prev_vdf_y) == str(parent_block.consensus_output())

    def verify_vdf(self, settings: Optional[PlatformConfig] = None) -> bool:
        active_settings = settings or build_platform_config()
        if self.vdf_y is None or self.vdf_proof is None or self.prev_vdf_y is None:
            return False

        z_hash = hash_data(self.prev_vdf_y, self.proposer_id, self.payload_hash)
        z_value = map_to_group(z_hash, active_settings.vdf_modulus_n)
        t_hash = hash_data(self.prev_vdf_y, self.proposer_id)
        delay_rounds = g_func(
            t_hash,
            min_delay_rounds=active_settings.min_delay_rounds,
            max_delay_rounds=active_settings.max_delay_rounds,
        )
        total_squarings = delay_rounds * active_settings.vdf_squarings_per_round
        modulus = active_settings.vdf_modulus_n
        return WesolowskiVDF.verify(modulus, z_value, self.vdf_y, total_squarings, self.vdf_proof)

    def verify_pow(self, settings: Optional[PlatformConfig] = None) -> bool:
        active_settings = settings or build_platform_config()
        if self.pow_hash is None or self.prev_vdf_y is None or self.nonce is None:
            return False

        y_star = hash_data(self.prev_vdf_y, self.proposer_id, self.payload_hash, self.nonce)
        return y_star == self.pow_hash and hash_to_int(y_star) < active_settings.pow_difficulty_phi

    def to_dict(self) -> dict[str, object]:
        return {
            "height": self.height,
            "parent_hash": self.parent_hash,
            "proposer_id": self.proposer_id,
            "prev_vdf_y": self.prev_vdf_y,
            "payload_hash": self.payload_hash,
            "vdf_y": self.vdf_y,
            "vdf_proof": self.vdf_proof,
            "round_generated": self.round_generated,
            "nonce": self.nonce,
            "pow_hash": self.pow_hash,
            "block_hash": self.block_hash,
        }

    def __str__(self) -> str:
        return f"Block(height={self.height}, proposer={self.proposer_id}, round={self.round_generated}, hash={self.block_hash[:8]})"


def build_genesis_block() -> Block:
    return Block(
        height=0,
        parent_hash="0" * 64,
        proposer_id=-1,
        prev_vdf_y=1,
        payload_hash="GENESIS_PAYLOAD",
        vdf_y=12345,
        vdf_proof=1,
        round_generated=0,
        nonce=0,
    )


class BaseMiner:
    should_restart_on_tip_change = True

    def __init__(self, miner_id: int, genesis_block: Block, settings: PlatformConfig) -> None:
        self.miner_id = miner_id
        self.settings = settings
        self.block_tree: Dict[str, Block] = {genesis_block.block_hash: genesis_block}
        self.pending_blocks: Dict[str, list[Block]] = {}
        self.local_tip = genesis_block.block_hash
        self.mining_task: Optional[Iterator[Optional[Block]]] = None
        self.mempool_counter = 0

    def verify_block(self, block: Block, parent_block: Block) -> bool:
        raise NotImplementedError

    def mine(self, parent_block: Block, current_round: int) -> Iterator[MiningTick]:
        raise NotImplementedError

    def process_received_blocks(self, blocks: list[Block], current_round: int) -> bool:
        for block in blocks:
            self._accept_or_buffer(block)

        best_tip = self.get_best_tip()
        tip_changed = best_tip != self.local_tip
        if tip_changed:
            self.local_tip = best_tip
            if self.should_restart_on_tip_change:
                self.mining_task = None
        return tip_changed

    def _accept_or_buffer(self, block: Block) -> None:
        if block.block_hash in self.block_tree:
            return

        parent_block = self.block_tree.get(block.parent_hash)
        if parent_block is None:
            self.pending_blocks.setdefault(block.parent_hash, []).append(block)
            return

        if not block.verify_parent_link(parent_block):
            return
        if not self.verify_block(block, parent_block):
            return

        self.block_tree[block.block_hash] = block
        self._drain_pending(block.block_hash)

    def _drain_pending(self, parent_hash: str) -> None:
        children = self.pending_blocks.pop(parent_hash, [])
        for child in children:
            self._accept_or_buffer(child)

    def get_best_tip(self) -> str:
        best_block = self.block_tree[self.local_tip]
        for candidate in self.block_tree.values():
            if candidate.height > best_block.height:
                best_block = candidate
                continue
            if candidate.height == best_block.height and candidate.block_hash < best_block.block_hash:
                best_block = candidate
        return best_block.block_hash

    def get_tip_block(self) -> Block:
        return self.block_tree[self.local_tip]

    def step(self, current_round: int) -> MinerStepReport:
        if self.mining_task is None:
            self.mining_task = self.mine(self.get_tip_block(), current_round)

        try:
            tick = next(self.mining_task)
        except StopIteration:
            self.mining_task = None
            return MinerStepReport(miner_id=self.miner_id, block=None, work_units=0, active_jobs=0)

        result = tick.block
        if result is None:
            return MinerStepReport(
                miner_id=self.miner_id,
                block=None,
                work_units=tick.work_units,
                active_jobs=tick.active_jobs,
            )

        self.mining_task = None
        result.round_generated = current_round
        result.refresh_hash()
        self.block_tree[result.block_hash] = result
        self.local_tip = result.block_hash
        return MinerStepReport(
            miner_id=self.miner_id,
            block=result,
            work_units=tick.work_units,
            active_jobs=tick.active_jobs,
        )


class PoVDMiner(BaseMiner):
    def verify_block(self, block: Block, parent_block: Block) -> bool:
        return block.verify_vdf(self.settings)

    def mine(self, parent_block: Block, current_round: int) -> Iterator[MiningTick]:
        self.mempool_counter += 1
        payload_hash = hash_data(f"Tx_{self.miner_id}_{self.mempool_counter}")
        previous_output = parent_block.consensus_output()
        z_hash = hash_data(previous_output, self.miner_id, payload_hash)
        z_value = map_to_group(z_hash, self.settings.vdf_modulus_n)
        t_hash = hash_data(previous_output, self.miner_id)
        delay_rounds = g_func(
            t_hash,
            min_delay_rounds=self.settings.min_delay_rounds,
            max_delay_rounds=self.settings.max_delay_rounds,
        )
        total_squarings = delay_rounds * self.settings.vdf_squarings_per_round

        y_value = z_value
        squarings_done = 0
        while squarings_done < total_squarings:
            budget = min(self.settings.vdf_squarings_per_round, total_squarings - squarings_done)
            for _ in range(budget):
                y_value = (y_value * y_value) % self.settings.vdf_modulus_n
            squarings_done += budget
            if squarings_done < total_squarings:
                yield MiningTick(block=None, work_units=budget)

        challenge = get_prime(hash_data(y_value, z_value, total_squarings), coprime_to=self.settings.vdf_phi)
        epsilon = pow(2, total_squarings, challenge)
        inverse = pow(challenge, -1, self.settings.vdf_phi)
        quotient = ((pow(2, total_squarings, self.settings.vdf_phi) - epsilon) * inverse) % self.settings.vdf_phi
        proof = pow(z_value, quotient, self.settings.vdf_modulus_n)

        yield MiningTick(
            block=Block(
                height=parent_block.height + 1,
                parent_hash=parent_block.block_hash,
                proposer_id=self.miner_id,
                prev_vdf_y=previous_output,
                payload_hash=payload_hash,
                vdf_y=y_value,
                vdf_proof=proof,
                round_generated=current_round,
            ),
            work_units=min(self.settings.vdf_squarings_per_round, total_squarings),
        )


class PoWMiner(BaseMiner):
    def verify_block(self, block: Block, parent_block: Block) -> bool:
        return block.verify_pow(self.settings)

    def mine(self, parent_block: Block, current_round: int) -> Iterator[MiningTick]:
        self.mempool_counter += 1
        payload_hash = hash_data(f"Tx_{self.miner_id}_{self.mempool_counter}")
        previous_output = parent_block.consensus_output()
        nonce = 0

        while True:
            for _ in range(self.settings.pow_hashes_per_round):
                pow_hash = hash_data(previous_output, self.miner_id, payload_hash, nonce)
                if hash_to_int(pow_hash) < self.settings.pow_difficulty_phi:
                    yield MiningTick(
                        block=Block(
                            height=parent_block.height + 1,
                            parent_hash=parent_block.block_hash,
                            proposer_id=self.miner_id,
                            prev_vdf_y=previous_output,
                            payload_hash=payload_hash,
                            vdf_y=None,
                            vdf_proof=None,
                            round_generated=current_round,
                            nonce=nonce,
                            pow_hash=pow_hash,
                        ),
                        work_units=self.settings.pow_hashes_per_round,
                    )
                    return
                nonce += 1
            yield MiningTick(block=None, work_units=self.settings.pow_hashes_per_round)


class BaselineVDFMiner(PoVDMiner):
    should_restart_on_tip_change = True


def _extend_propagation_vector(propagation_vector: tuple[float, ...], extra_steps: int) -> tuple[float, ...]:
    if not propagation_vector or extra_steps <= 0:
        return propagation_vector

    base = tuple(propagation_vector)
    if base[-1] < 1.0:
        base = base + (1.0,)

    if base[-1] != 1.0:
        return base

    if len(base) == 1:
        near_full = 0.99
        return (near_full,) * extra_steps + base

    result = list(base)
    for _ in range(extra_steps):
        prev = result[-2]
        inserted = min(0.99, max(prev, 0.0) + 0.05)
        result.insert(-1, inserted)
    return tuple(result)


@dataclass(order=True)
class NetworkEvent:
    deliver_round: int
    receiver_id: int
    block: Block = field(compare=False)


class Network:
    def __init__(self, num_miners: int, propagation_vector: tuple[float, ...]) -> None:
        self.num_miners = num_miners
        self.propagation_vector = propagation_vector
        self.event_queue: list[NetworkEvent] = []

    def broadcast(self, sender_id: int, block: Block, current_round: int) -> None:
        targets = [miner_id for miner_id in range(self.num_miners) if miner_id != sender_id]
        randomizer = hash_to_int(hash_data(block.block_hash, sender_id, current_round))
        ordered_targets = sorted(targets, key=lambda miner_id: hash_data(randomizer, miner_id))

        current_target_idx = 0
        for delay, weight in enumerate(self.propagation_vector):
            expected_total_count = int(self.num_miners * weight)
            current_total_count = 1 + current_target_idx
            needed = expected_total_count - current_total_count

            for _ in range(max(0, needed)):
                if current_target_idx >= len(ordered_targets):
                    break
                self.event_queue.append(
                    NetworkEvent(
                        deliver_round=current_round + delay,
                        receiver_id=ordered_targets[current_target_idx],
                        block=block,
                    )
                )
                current_target_idx += 1

        final_delay = len(self.propagation_vector)
        while current_target_idx < len(ordered_targets):
            self.event_queue.append(
                NetworkEvent(
                    deliver_round=current_round + final_delay,
                    receiver_id=ordered_targets[current_target_idx],
                    block=block,
                )
            )
            current_target_idx += 1

        self.event_queue.sort()

    def deliver(self, current_round: int) -> dict[int, list[Block]]:
        deliveries: dict[int, list[Block]] = {}
        remaining_events: list[NetworkEvent] = []

        for event in self.event_queue:
            if event.deliver_round <= current_round:
                deliveries.setdefault(event.receiver_id, []).append(event.block)
            else:
                remaining_events.append(event)

        self.event_queue = remaining_events
        return deliveries


@dataclass
class PlatformResult:
    mode: str
    rounds: int
    total_blocks: int
    main_chain_height: int
    orphan_blocks: int
    fork_rate: float
    distinct_chain_tips: int
    consensus_reached: bool
    duration_seconds: float
    average_active_threads: float
    peak_active_threads: int
    average_power_load: float
    peak_power_load: float
    thread_rounds_per_main_block: float
    total_work_units: int
    work_unit_name: str

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "rounds": self.rounds,
            "total_blocks": self.total_blocks,
            "main_chain_height": self.main_chain_height,
            "height": self.main_chain_height,
            "orphan_blocks": self.orphan_blocks,
            "fork_rate": self.fork_rate,
            "distinct_chain_tips": self.distinct_chain_tips,
            "consensus_reached": self.consensus_reached,
            "duration_seconds": self.duration_seconds,
            "average_active_threads": self.average_active_threads,
            "peak_active_threads": self.peak_active_threads,
            "average_power_load": self.average_power_load,
            "peak_power_load": self.peak_power_load,
            "thread_rounds_per_main_block": self.thread_rounds_per_main_block,
            "total_work_units": self.total_work_units,
            "work_unit_name": self.work_unit_name,
        }


class Platform:
    def __init__(self, mode: str, settings: Optional[PlatformConfig] = None) -> None:
        normalized_mode = mode.lower()
        if normalized_mode not in {"povd", "pow", "vdf_baseline"}:
            raise ValueError("mode must be one of 'povd', 'pow', or 'vdf_baseline'")

        self.mode = normalized_mode
        self.settings = settings or build_platform_config()
        self.genesis_block = build_genesis_block()
        miner_cls = {
            "povd": PoVDMiner,
            "pow": PoWMiner,
            "vdf_baseline": BaselineVDFMiner,
        }[self.mode]
        self.miners = [miner_cls(miner_id, self.genesis_block, self.settings) for miner_id in range(self.settings.num_miners)]
        base_vector = self.settings.propagation_vector
        extra_delay_steps = 0
        if self.mode == "vdf_baseline":
            extra_delay_steps = 2
        elif self.mode == "pow":
            extra_delay_steps = 3
        self.network = Network(self.settings.num_miners, _extend_propagation_vector(base_vector, extra_delay_steps))

    def run(self, verbose: bool = True, round_limit: Optional[int] = None) -> PlatformResult:
        total_blocks_mined = 0
        current_round = 0
        max_height_reached = 0
        safety_limit = round_limit or (self.settings.target_block_height * self.settings.max_round_multiplier)
        total_work_units = 0
        total_active_threads = 0
        peak_active_threads = 0
        total_power_load = 0.0
        peak_power_load = 0.0
        per_thread_capacity = self.settings.pow_hashes_per_round if self.mode == "pow" else self.settings.vdf_squarings_per_round
        work_unit_name = "hashes" if self.mode == "pow" else "squarings"

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.settings.num_miners, thread_name_prefix="miner") as executor:
            while max_height_reached < self.settings.target_block_height:
                current_round += 1
                deliveries = self.network.deliver(current_round)
                round_new_blocks: list[tuple[int, Block]] = []

                for miner in self.miners:
                    miner.process_received_blocks(deliveries.get(miner.miner_id, []), current_round)

                reports = [future.result() for future in [executor.submit(miner.step, current_round) for miner in self.miners]]

                round_active_threads = 0
                round_power_load = 0.0
                for report in reports:
                    total_work_units += report.work_units
                    if report.work_units > 0:
                        round_active_threads += 1
                        round_power_load += min(1.0, report.work_units / per_thread_capacity) / self.settings.num_miners
                    if report.block is None:
                        continue
                    round_new_blocks.append((report.miner_id, report.block))
                    total_blocks_mined += 1
                    if verbose:
                        print(
                            f"[Round {current_round}] {self.mode.upper()} miner {report.miner_id} mined "
                            f"height {report.block.height} ({report.block.block_hash[:8]})"
                        )

                total_active_threads += round_active_threads
                peak_active_threads = max(peak_active_threads, round_active_threads)
                total_power_load += round_power_load
                peak_power_load = max(peak_power_load, round_power_load)

                for sender_id, block in round_new_blocks:
                    self.network.broadcast(sender_id, block, current_round)

                max_height_reached = max(miner.get_tip_block().height for miner in self.miners)
                if current_round > safety_limit:
                    break

        duration_seconds = time.time() - start_time
        distinct_chain_tips = len({miner.get_tip_block().block_hash for miner in self.miners})
        orphan_blocks = max(0, total_blocks_mined - max_height_reached)
        fork_rate = (orphan_blocks / total_blocks_mined) if total_blocks_mined else 0.0
        average_active_threads = (total_active_threads / current_round) if current_round else 0.0
        average_power_load = (total_power_load / current_round) if current_round else 0.0
        thread_rounds_per_main_block = (total_active_threads / max_height_reached) if max_height_reached else 0.0

        return PlatformResult(
            mode=self.mode.upper(),
            rounds=current_round,
            total_blocks=total_blocks_mined,
            main_chain_height=max_height_reached,
            orphan_blocks=orphan_blocks,
            fork_rate=fork_rate,
            distinct_chain_tips=distinct_chain_tips,
            consensus_reached=distinct_chain_tips == 1,
            duration_seconds=duration_seconds,
            average_active_threads=average_active_threads,
            peak_active_threads=peak_active_threads,
            average_power_load=average_power_load,
            peak_power_load=peak_power_load,
            thread_rounds_per_main_block=thread_rounds_per_main_block,
            total_work_units=total_work_units,
            work_unit_name=work_unit_name,
        )


def run_platform(
    mode: str,
    settings: Optional[PlatformConfig] = None,
    verbose: bool = True,
    round_limit: Optional[int] = None,
) -> dict[str, object]:
    return Platform(mode=mode, settings=settings).run(verbose=verbose, round_limit=round_limit).to_dict()

def compare_platforms(
    settings: Optional[PlatformConfig] = None,
    verbose: bool = True,
    round_limit: Optional[int] = None,
) -> dict[str, dict[str, object]]:
    active_settings = settings or build_platform_config()
    return {
        "povd": run_platform("povd", settings=active_settings, verbose=verbose, round_limit=round_limit),
        "pow": run_platform("pow", settings=active_settings, verbose=verbose, round_limit=round_limit),
        "vdf_baseline": run_platform("vdf_baseline", settings=active_settings, verbose=verbose, round_limit=round_limit),
    }


def format_platform_result(result: dict[str, object]) -> str:
    return "\n".join(
        [
            f"Total Blocks: {result['total_blocks']}",
            f"Main Height: {result['main_chain_height']}",
            f"Orphan Blocks: {result['orphan_blocks']}",
            f"Fork Rate: {result['fork_rate']:.4f}",
        ]
    )


def format_comparison(results: dict[str, dict[str, object]]) -> str:
    povd_result = results["povd"]
    pow_result = results["pow"]
    baseline_result = results["vdf_baseline"]
    lines = [
        "Metric           | PoVD            | PoW             | VDF_Baseline",
        "-" * 74,
        f"Total Blocks     | {povd_result['total_blocks']:<15} | {pow_result['total_blocks']:<15} | {baseline_result['total_blocks']:<15}",
        f"Main Height      | {povd_result['main_chain_height']:<15} | {pow_result['main_chain_height']:<15} | {baseline_result['main_chain_height']:<15}",
        f"Orphan Blocks    | {povd_result['orphan_blocks']:<15} | {pow_result['orphan_blocks']:<15} | {baseline_result['orphan_blocks']:<15}",
        f"Fork Rate        | {povd_result['fork_rate']:<15.4f} | {pow_result['fork_rate']:<15.4f} | {baseline_result['fork_rate']:<15.4f}",
    ]
    return "\n".join(lines)


MiningPlatform = Platform
MiningPlatformConfig = PlatformConfig
MiningPlatformResult = PlatformResult
run_mining_platform = run_platform
compare_mining_modes = compare_platforms
format_mining_result = format_platform_result


__all__ = [
    "BaseMiner",
    "Block",
    "Network",
    "NetworkEvent",
    "BaselineVDFMiner",
    "MinerStepReport",
    "MiningTick",
    "PoVDMiner",
    "PoWMiner",
    "MiningPlatform",
    "MiningPlatformConfig",
    "MiningPlatformResult",
    "Platform",
    "PlatformConfig",
    "PlatformResult",
    "build_platform_config",
    "build_genesis_block",
    "compare_mining_modes",
    "compare_platforms",
    "format_comparison",
    "format_mining_result",
    "format_platform_result",
    "run_mining_platform",
    "run_platform",
]
