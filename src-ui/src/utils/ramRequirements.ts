/**
 * Recalculated Room Stable RAM requirement formula shared with
 * `core-engine/shard_engine/memory_manager.py#calculate_room_ram_requirements`.
 */
import type { CatalogModelItem } from "../types/node";

export type RamReadinessStatus =
  | "solo_stable"
  | "solo_tight"
  | "swarm_needed"
  | "insufficient";

export interface RoomRamEstimate {
  modelSizeGb: number;
  totalLayers: number;
  nodeCount: number;
  perNodeOverheadGb: number;
  shardRoundingSlackGb: number;
  safetyMarginGb: number;
  minRoomRamGb: number;
  stableRoomRamGb: number;
  recommendedSliderGb: number;
}

export interface NodeContributionAssessment {
  status: RamReadinessStatus;
  modelSizeGb: number;
  sharedRamGb: number;
  totalLayers: number;
  perNodeOverheadGb: number;
  layersFit: number;
  estimatedCoveragePct: number;
  soloMinGb: number;
  soloStableGb: number;
  soloSliderGb: number;
  twoNodeStableGb: number;
  estimatedPeersNeeded: number | null;
  targetPeers: number;
  targetRoomStableGb: number;
  remainingRamForTargetRoomGb: number;
}

function roundTo(value: number, decimals: number): number {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}

function ceilToHalfGb(value: number): number {
  return Math.ceil(value * 2 - 1e-9) / 2;
}

/** Format GB values cleanly: `7` -> `"7 GB"`, `7.5` -> `"7.5 GB"`. */
export function formatGb(value: number): string {
  const rounded = roundTo(value, 1);
  return `${Number.isInteger(rounded) ? rounded.toFixed(0) : rounded.toFixed(1)} GB`;
}

/** Estimate transformer layer count from quantized `.gguf` size in GB. */
export function estimateTotalLayers(modelSizeGb: number): number {
  if (modelSizeGb <= 2.5) {
    return 24;
  }
  if (modelSizeGb < 9.0) {
    return 32;
  }
  if (modelSizeGb < 18.0) {
    return 40;
  }
  return 64;
}

/** Per-node fixed KV-cache + runtime + scratch buffer overhead in GB. */
export function getPerNodeOverheadGb(modelSizeGb: number): number {
  return modelSizeGb <= 1.5 ? 0.35 : 0.6;
}

/**
 * Calculate minimum and stable physical room RAM for a 1-node or N-node swarm.
 * Mirrors `calculate_room_ram_requirements` in `shard_engine/memory_manager.py`.
 */
export function calculateRoomRamEstimate(
  modelSizeGb: number,
  nodeCount: number = 1,
  totalLayers?: number
): RoomRamEstimate {
  const safeSize = Number.isFinite(modelSizeGb) && modelSizeGb > 0 ? modelSizeGb : 5.0;
  const safeNodes = Math.max(1, Math.floor(nodeCount));
  const resolvedLayers =
    typeof totalLayers === "number" && Number.isFinite(totalLayers) && totalLayers > 0
      ? Math.floor(totalLayers)
      : estimateTotalLayers(safeSize);

  const perNodeOverheadGb = getPerNodeOverheadGb(safeSize);
  const perLayerGb = safeSize / resolvedLayers;
  const shardRoundingSlackGb = (safeNodes - 1) * perLayerGb;
  const safetyMarginGb = Math.max(0.2, 0.1 * safeSize);

  const rawMin = safeSize + safeNodes * perNodeOverheadGb + shardRoundingSlackGb;
  const rawStable = rawMin + safetyMarginGb;
  const stableRoomRamGb = ceilToHalfGb(rawStable);

  return {
    modelSizeGb: roundTo(safeSize, 2),
    totalLayers: resolvedLayers,
    nodeCount: safeNodes,
    perNodeOverheadGb: roundTo(perNodeOverheadGb, 2),
    shardRoundingSlackGb: roundTo(shardRoundingSlackGb, 2),
    safetyMarginGb: roundTo(safetyMarginGb, 2),
    minRoomRamGb: roundTo(rawMin, 2),
    stableRoomRamGb,
    recommendedSliderGb: Math.max(2, Math.ceil(stableRoomRamGb - 1e-9)),
  };
}

/**
 * Evaluate how well the user's `sharedRamGb` covers the selected model on its own
 * vs inside a multi-peer room.
 */
export function assessNodeRamContribution(
  modelSizeGb: number,
  sharedRamGb: number,
  totalLayers?: number
): NodeContributionAssessment {
  const solo = calculateRoomRamEstimate(modelSizeGb, 1, totalLayers);
  const twoNode = calculateRoomRamEstimate(modelSizeGb, 2, solo.totalLayers);
  const perLayerGb = solo.modelSizeGb / solo.totalLayers;
  const usableGb = Math.max(0, sharedRamGb - solo.perNodeOverheadGb);
  const layersFit = Math.max(
    0,
    Math.min(solo.totalLayers, Math.floor(usableGb / perLayerGb))
  );
  const estimatedCoveragePct = Math.min(
    100,
    Math.max(0, Math.round((layersFit / solo.totalLayers) * 100))
  );

  const estimatedPeersNeeded =
    layersFit >= solo.totalLayers
      ? 1
      : layersFit > 0
        ? Math.ceil(solo.totalLayers / layersFit)
        : null;

  const targetPeers =
    estimatedPeersNeeded !== null ? Math.max(2, estimatedPeersNeeded) : 2;
  const targetRoom = calculateRoomRamEstimate(
    solo.modelSizeGb,
    targetPeers,
    solo.totalLayers
  );
  const remainingRamForTargetRoomGb = Math.max(
    0,
    roundTo(targetRoom.stableRoomRamGb - sharedRamGb, 1)
  );

  let status: RamReadinessStatus;
  if (sharedRamGb >= solo.stableRoomRamGb) {
    status = "solo_stable";
  } else if (layersFit >= solo.totalLayers) {
    status = "solo_tight";
  } else if (layersFit >= 1) {
    status = "swarm_needed";
  } else {
    status = "insufficient";
  }

  return {
    status,
    modelSizeGb: solo.modelSizeGb,
    sharedRamGb,
    totalLayers: solo.totalLayers,
    perNodeOverheadGb: solo.perNodeOverheadGb,
    layersFit,
    estimatedCoveragePct,
    soloMinGb: solo.minRoomRamGb,
    soloStableGb: solo.stableRoomRamGb,
    soloSliderGb: solo.recommendedSliderGb,
    twoNodeStableGb: twoNode.stableRoomRamGb,
    estimatedPeersNeeded,
    targetPeers,
    targetRoomStableGb: targetRoom.stableRoomRamGb,
    remainingRamForTargetRoomGb,
  };
}

function normalizeModelIdentifier(value: string): string {
  const trimmed = value.trim().toLowerCase();
  const tail = trimmed.includes("/") ? trimmed.slice(trimmed.lastIndexOf("/") + 1) : trimmed;
  return tail.replace(/[-_]?gguf$/i, "");
}

/** Match a room's or input's model string to a known catalog entry for RAM hints. */
export function resolveCatalogModelForRoom(
  modelIdentifier: string,
  catalogModels: readonly CatalogModelItem[]
): CatalogModelItem | undefined {
  const trimmed = modelIdentifier.trim();
  if (!trimmed || catalogModels.length === 0) {
    return undefined;
  }
  const lowered = trimmed.toLowerCase();
  const exact = catalogModels.find(
    (item) =>
      item.repo_id === trimmed ||
      item.id === trimmed ||
      item.repo_id.toLowerCase() === lowered ||
      item.id.toLowerCase() === lowered ||
      item.name.toLowerCase() === lowered
  );
  if (exact) {
    return exact;
  }
  const normalizedTarget = normalizeModelIdentifier(trimmed);
  if (!normalizedTarget) {
    return undefined;
  }
  return catalogModels.find(
    (item) =>
      normalizeModelIdentifier(item.repo_id) === normalizedTarget ||
      normalizeModelIdentifier(item.id) === normalizedTarget ||
      normalizeModelIdentifier(item.name) === normalizedTarget
  );
}
