import { describe, expect, it } from "vitest";
import type { CatalogModelItem } from "../types/node";
import {
  assessNodeRamContribution,
  calculateRoomRamEstimate,
  estimateTotalLayers,
  formatGb,
  getPerNodeOverheadGb,
  resolveCatalogModelForRoom,
} from "../utils/ramRequirements";

describe("ramRequirements formula", () => {
  it("computes consistent Solo and multi-node Room Stable RAM for 5.5 GB DeepSeek", () => {
    expect(estimateTotalLayers(5.5)).toBe(32);
    expect(getPerNodeOverheadGb(5.5)).toBe(0.6);

    const solo = calculateRoomRamEstimate(5.5, 1);
    expect(solo.minRoomRamGb).toBe(6.1);
    expect(solo.stableRoomRamGb).toBe(7.0);
    expect(solo.recommendedSliderGb).toBe(7);

    const twoNodes = calculateRoomRamEstimate(5.5, 2);
    expect(twoNodes.minRoomRamGb).toBe(6.87);
    expect(twoNodes.stableRoomRamGb).toBe(7.5);

    const threeNodes = calculateRoomRamEstimate(5.5, 3);
    expect(threeNodes.minRoomRamGb).toBe(7.64);
    expect(threeNodes.stableRoomRamGb).toBe(8.5);
  });

  it("uses reduced 0.35 GB overhead for tiny <= 1.5 GB models", () => {
    expect(getPerNodeOverheadGb(0.6)).toBe(0.35);
    const est = calculateRoomRamEstimate(0.6, 2);
    expect(est.perNodeOverheadGb).toBe(0.35);
    expect(est.stableRoomRamGb).toBeLessThanOrEqual(2);
  });

  it("scales monotonically for large 20 GB models", () => {
    const two = calculateRoomRamEstimate(20.0, 2);
    const four = calculateRoomRamEstimate(20.0, 4);
    expect(four.minRoomRamGb).toBeGreaterThan(two.minRoomRamGb);
    expect(four.stableRoomRamGb).toBeGreaterThan(two.stableRoomRamGb);
  });

  it("assesses node RAM contribution across all 4 readiness tiers", () => {
    // 5.5 GB DeepSeek:
    // - 7 GB => solo_stable
    const soloStable = assessNodeRamContribution(5.5, 7);
    expect(soloStable.status).toBe("solo_stable");
    expect(soloStable.estimatedCoveragePct).toBe(100);
    expect(soloStable.estimatedPeersNeeded).toBe(1);

    // - 6.2 GB => 32 layers fit (6.2 - 0.6 = 5.6 >= 5.5), below 7.0 GB stable => solo_tight
    const soloTight = assessNodeRamContribution(5.5, 6.2);
    expect(soloTight.status).toBe("solo_tight");
    expect(soloTight.estimatedCoveragePct).toBe(100);
    expect(soloTight.soloSliderGb).toBe(7);

    // - 4 GB => 19/32 layers (~59%) => swarm_needed (2 peers -> 7.5 GB stable, +3.5 GB needed)
    const swarmNeeded = assessNodeRamContribution(5.5, 4);
    expect(swarmNeeded.status).toBe("swarm_needed");
    expect(swarmNeeded.layersFit).toBe(19);
    expect(swarmNeeded.targetPeers).toBe(2);
    expect(swarmNeeded.targetRoomStableGb).toBe(7.5);
    expect(swarmNeeded.remainingRamForTargetRoomGb).toBe(3.5);

    // - 1 GB on 20 GB model (usable 0.4 GB < 0.3125? wait: 0.4/0.3125=1 layer, test 2 GB on 130 GB => 0 layers)
    const insufficient = assessNodeRamContribution(130, 1);
    expect(insufficient.status).toBe("insufficient");
    expect(insufficient.layersFit).toBe(0);
    expect(insufficient.estimatedPeersNeeded).toBeNull();
  });

  it("resolves catalog models by repo_id, id, and case-insensitive name and formats GB cleanly", () => {
    const catalog: CatalogModelItem[] = [
      {
        id: "deepseek-r1-distill-llama-8b",
        name: "DeepSeek-R1 Distill Llama 8B",
        repo_id: "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF",
        filename: "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
        estimated_ram_gb: 5.5,
        enabled: true,
        task: "text",
      },
    ];

    expect(
      resolveCatalogModelForRoom("bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF", catalog)?.id
    ).toBe("deepseek-r1-distill-llama-8b");
    expect(resolveCatalogModelForRoom("deepseek-r1-distill-llama-8b", catalog)?.id).toBe(
      "deepseek-r1-distill-llama-8b"
    );
    expect(resolveCatalogModelForRoom("DEEPSEEK-R1 DISTILL LLAMA 8B", catalog)?.id).toBe(
      "deepseek-r1-distill-llama-8b"
    );
    expect(resolveCatalogModelForRoom("unknown/model", catalog)).toBeUndefined();

    expect(formatGb(7.0)).toBe("7 GB");
    expect(formatGb(7.5)).toBe("7.5 GB");
  });
});
