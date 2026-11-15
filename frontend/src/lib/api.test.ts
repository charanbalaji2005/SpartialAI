import { describe, expect, it, vi, afterEach } from "vitest";
import { api, ApiError, STATUS_LABEL } from "./api";

afterEach(() => vi.unstubAllGlobals());

describe("api client", () => {
  it("builds export and asset URLs", () => {
    expect(api.exportUrl("abc", "glb")).toBe("/api/export/abc/glb");
    expect(api.assetUrl("abc.assets.obj_01_chair.glb")).toBe("/api/assets/abc.assets.obj_01_chair.glb");
  });

  it("surfaces backend error detail", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail: "Use a JPG, PNG or WEBP image." }), { status: 422 })));
    await expect(api.projects()).rejects.toMatchObject({ message: "Use a JPG, PNG or WEBP image.", status: 422 });
  });

  it("explains an unreachable backend", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    const err = await api.health().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.message).toMatch(/start_spatialai\.bat/);
  });

  it("labels every geometry status", () => {
    expect(Object.keys(STATUS_LABEL).sort()).toEqual(["generated", "inferred", "observed"]);
  });
});
