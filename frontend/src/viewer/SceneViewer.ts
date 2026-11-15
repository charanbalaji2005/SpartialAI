import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { GLTFExporter } from "three/examples/jsm/exporters/GLTFExporter.js";
import type { GeometryStatus, SceneMeta, SceneNode } from "../lib/api";

export type ShadingMode = "photo" | "lit" | "clay" | "truth" | "check";
export type ViewPreset = "front" | "back" | "left" | "right" | "top" | "bottom";

const PRESET_DIRS: Record<ViewPreset, [number, number, number]> = {
  front: [0, 0, 1], back: [0, 0, -1], left: [-1, 0, 0], right: [1, 0, 0],
  top: [0, 1, 0.0001], bottom: [0, -1, 0.0001],
};

/** Mesh-check material: front faces shaded in the part's status colour, back faces solid red.
 *  On a closed, outward-facing mesh no red is ever visible, so holes and flipped normals stand out. */
function checkMaterial(color: number, wireframe: boolean): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    side: THREE.DoubleSide,
    wireframe,
    uniforms: { uColor: { value: new THREE.Color(color) } },
    vertexShader: `varying vec3 vN;
      void main() { vN = normalize(normalMatrix * normal); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `varying vec3 vN; uniform vec3 uColor;
      void main() {
        float l = 0.35 + 0.65 * abs(normalize(vN).z);
        gl_FragColor = gl_FrontFacing ? vec4(uColor * l, 1.0) : vec4(0.92, 0.14, 0.14, 1.0);
      }`,
  });
}
export type ToolMode = "select" | "translate" | "rotate" | "scale";
export interface Transform { position: number[]; rotation: number[]; scale: number[] }

export const STATUS_COLORS: Record<GeometryStatus, number> = {
  observed: 0x4fb3c8,
  inferred: 0xc07fd8,
  generated: 0x8fcf5a,
};

interface Entry { mesh: THREE.Mesh; meta: SceneNode; original: THREE.Material | THREE.Material[] }

const r2d = THREE.MathUtils.radToDeg;
const d2r = THREE.MathUtils.degToRad;

/** Imperative Three.js engine; React drives it through this API. */
export class SceneViewer {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  private persp: THREE.PerspectiveCamera;
  private ortho: THREE.OrthographicCamera;
  private camera: THREE.Camera;
  private controls: OrbitControls;
  private gizmo: TransformControls;
  private gizmoHelper: THREE.Object3D;
  private grid: THREE.GridHelper;
  private axes: THREE.AxesHelper;
  private key = new THREE.DirectionalLight(0xffffff, 1.6);
  private hemi = new THREE.HemisphereLight(0xffffff, 0x404850, 1.2);
  private root: THREE.Group | null = null;
  private entries = new Map<string, Entry>();
  private pivots = new Map<string, THREE.Group>();
  private home = new Map<string, { position: THREE.Vector3; quaternion: THREE.Quaternion; scale: THREE.Vector3 }>();
  private selectionBox: THREE.Box3Helper | null = null;
  private selectedGroup: string | null = null;
  private view = { position: new THREE.Vector3(), target: new THREE.Vector3(), vfov: 50 };
  private frame = 0;
  private resizeObs: ResizeObserver;
  private down = { x: 0, y: 0 };
  private dragged = false;
  private axisCanvas: HTMLCanvasElement;
  private tool: ToolMode = "translate";
  shading: ShadingMode = "photo";
  wireframe = false;

  onSelect: (group: string | null) => void = () => {};
  onTransform: (group: string, t: Transform) => void = () => {};

  constructor(private container: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.setClearColor(0x1c1c1c);
    container.appendChild(this.renderer.domElement);

    this.persp = new THREE.PerspectiveCamera(50, 1, 0.01, 2000);
    this.ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, -1000, 2000);
    this.camera = this.persp;
    this.controls = new OrbitControls(this.persp, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.screenSpacePanning = true;
    this.controls.minPolarAngle = 0.01;
    this.controls.maxPolarAngle = Math.PI - 0.01;
    this.controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.PAN,
      RIGHT: THREE.MOUSE.ROTATE,
    };

    this.gizmo = new TransformControls(this.persp, this.renderer.domElement);
    this.gizmo.size = 0.9;
    this.gizmoHelper = this.gizmo.getHelper();
    this.scene.add(this.gizmoHelper);
    this.gizmo.addEventListener("dragging-changed", (e) => {
      this.controls.enabled = !(e as unknown as { value: boolean }).value;
      if ((e as unknown as { value: boolean }).value) this.dragged = true;
    });
    this.gizmo.addEventListener("objectChange", () => {
      if (this.selectedGroup) { this.updateSelectionBox(); this.onTransform(this.selectedGroup, this.getTransform(this.selectedGroup)!); }
    });

    this.grid = new THREE.GridHelper(40, 40, 0x5a5a5a, 0x333333);
    this.axes = new THREE.AxesHelper(0.6);
    this.axes.visible = false;
    this.scene.add(this.grid, this.axes, this.hemi, this.key);

    this.axisCanvas = document.createElement("canvas");
    this.axisCanvas.className = "axis-indicator";
    this.axisCanvas.width = this.axisCanvas.height = 120;
    container.appendChild(this.axisCanvas);

    const el = this.renderer.domElement;
    el.addEventListener("contextmenu", (e) => e.preventDefault());
    el.addEventListener("pointerdown", (e) => { this.down = { x: e.clientX, y: e.clientY }; this.dragged = false; });
    el.addEventListener("pointerup", (e) => {
      if (!this.dragged && Math.hypot(e.clientX - this.down.x, e.clientY - this.down.y) < 4) {
        if (e.button === 0) this.pick(e);
      }
    });
    el.addEventListener("dblclick", (e) => {
      this.pick(e);
      this.focusSelection();
    });

    this.resizeObs = new ResizeObserver(() => this.resize());
    this.resizeObs.observe(container);
    this.resize();
    const loop = () => {
      this.frame = requestAnimationFrame(loop);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
      this.drawAxes();
    };
    loop();
  }

  async load(meta: SceneMeta): Promise<void> {
    this.clear();
    const gltf = await new GLTFLoader().loadAsync(meta.glb_url + `?v=${Date.now()}`);
    this.root = gltf.scene;
    this.scene.add(this.root);
    for (const node of meta.nodes) {
      const obj = this.root.getObjectByName(node.node);
      const mesh = obj && ((obj as THREE.Mesh).isMesh ? (obj as THREE.Mesh) : (obj.children.find((c) => (c as THREE.Mesh).isMesh) as THREE.Mesh | undefined));
      if (!mesh) continue;
      if (!mesh.geometry.attributes.normal) mesh.geometry.computeVertexNormals();
      mesh.userData.key = SceneViewer.keyOf(node);
      mesh.visible = node.default_visible;
      this.entries.set(node.node, { mesh, meta: node, original: mesh.material });
    }
    // Each actor (object group or individual region) gets a pivot at its centre so it moves, rotates and scales about itself.
    const keys = new Set([...this.entries.values()].map((e) => SceneViewer.keyOf(e.meta)));
    for (const k of keys) {
      const pivot = new THREE.Group();
      pivot.name = `pivot_${k}`;
      pivot.position.fromArray(this.keyCenter(k));
      this.root.add(pivot);
      for (const e of this.entries.values()) if (SceneViewer.matches(e.meta, k)) {
        e.mesh.updateMatrixWorld(true);
        pivot.attach(e.mesh);
      }
      this.pivots.set(k, pivot);
      this.home.set(k, { position: pivot.position.clone(), quaternion: pivot.quaternion.clone(), scale: pivot.scale.clone() });
    }

    const [r, g, b] = meta.lighting.key_color;
    this.key.color.setRGB(r, g, b, THREE.SRGBColorSpace);
    const [dx, dy, dz] = meta.lighting.key_direction;
    this.key.position.set(dx * 5, dy * 5 + 1, dz * 5);
    this.hemi.intensity = 0.6 + meta.lighting.ambient_intensity;

    this.view = { position: new THREE.Vector3(0, 0, 0.001), target: new THREE.Vector3(...(meta.target as [number, number, number])), vfov: meta.camera.vfov_deg };
    const box = new THREE.Box3().setFromObject(this.root);
    this.grid.position.y = box.min.y;
    this.grid.scale.setScalar(Math.max(1, box.getSize(new THREE.Vector3()).length() / 20));
    this.applyShading();
    this.resetCamera();
  }

  // ---------------------------------------------------------------- camera
  resetCamera(): void {
    this.persp.fov = this.view.vfov;
    this.persp.position.copy(this.view.position);
    this.controls.target.copy(this.view.target);
    this.persp.updateProjectionMatrix();
    this.controls.enableDamping = false;
    this.controls.update();
    this.controls.enableDamping = true;
    this.syncOrtho();
  }

  setOrthographic(on: boolean): void {
    this.camera = on ? this.ortho : this.persp;
    this.controls.object = this.camera;
    this.gizmo.camera = this.camera;
    if (on) { this.ortho.position.copy(this.persp.position); this.ortho.quaternion.copy(this.persp.quaternion); this.syncOrtho(); }
    else this.persp.position.copy(this.ortho.position);
    this.controls.update();
  }

  setCameraSpeed(speed: number): void {
    this.controls.rotateSpeed = speed;
    this.controls.panSpeed = speed;
    this.controls.zoomSpeed = speed;
  }

  focusSelection(): void {
    const meshes = this.selectedMeshes().filter((m) => m.visible);
    if (!meshes.length) return;
    const box = new THREE.Box3();
    meshes.forEach((m) => { m.updateMatrixWorld(true); box.expandByObject(m); });
    const center = box.getCenter(new THREE.Vector3());
    const size = Math.max(0.5, box.getSize(new THREE.Vector3()).length());
    const dir = this.camera.position.clone().sub(this.controls.target).normalize();
    if (dir.lengthSq() < 0.5) dir.set(0, 0.2, 1).normalize();
    this.controls.target.copy(center);
    this.camera.position.copy(center.clone().add(dir.multiplyScalar(size * 1.4 + 0.2)));
    this.controls.update();
    this.syncOrtho();
  }

  /** Orbit to a fixed side of the scene (front = the photo's side), framing everything visible. */
  viewPreset(preset: ViewPreset): void {
    const box = new THREE.Box3();
    for (const e of this.entries.values()) if (e.mesh.visible) { e.mesh.updateMatrixWorld(true); box.expandByObject(e.mesh); }
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const radius = box.getSize(new THREE.Vector3()).length() / 2;
    const dist = (radius / Math.tan(d2r(this.persp.fov / 2))) * 1.1;
    const dir = new THREE.Vector3(...PRESET_DIRS[preset]).normalize();
    this.controls.target.copy(center);
    this.camera.position.copy(center.clone().add(dir.multiplyScalar(dist)));
    this.controls.enableDamping = false;
    this.controls.update();
    this.controls.enableDamping = true;
    this.syncOrtho();
  }

  // ---------------------------------------------------------------- display
  setShading(mode: ShadingMode): void { this.shading = mode; this.applyShading(); }
  setWireframe(on: boolean): void { this.wireframe = on; this.applyShading(); }
  setGrid(on: boolean): void { this.grid.visible = on; }
  setAxes(on: boolean): void { this.axes.visible = on; }
  setVisible(nodes: string[], visible: boolean): void {
    for (const n of nodes) { const e = this.entries.get(n); if (e) e.mesh.visible = visible; }
    this.updateSelectionBox();
  }

  // ---------------------------------------------------------------- selection + transform
  setTool(tool: ToolMode): void { this.tool = tool; this.attachGizmo(); }
  setSpace(space: "world" | "local"): void { this.gizmo.setSpace(space); }
  setSnap(translate: number | null, rotateDeg: number | null, scale: number | null): void {
    this.gizmo.setTranslationSnap(translate);
    this.gizmo.setRotationSnap(rotateDeg ? d2r(rotateDeg) : null);
    this.gizmo.setScaleSnap(scale);
  }

  /** Select by key: an object's group id (all its parts) or a single region node name. */
  select(key: string | null): void {
    this.selectedGroup = key && [...this.entries.values()].some((e) => SceneViewer.matches(e.meta, key)) ? key : null;
    this.attachGizmo();
    this.updateSelectionBox();
  }

  getTransform(group: string): Transform | null {
    const p = this.pivots.get(group);
    if (!p) return null;
    const e = new THREE.Euler().setFromQuaternion(p.quaternion, "XYZ");
    return { position: p.position.toArray(), rotation: [r2d(e.x), r2d(e.y), r2d(e.z)], scale: p.scale.toArray() };
  }

  setTransform(group: string, t: Transform): void {
    const p = this.pivots.get(group);
    if (!p) return;
    p.position.fromArray(t.position);
    p.rotation.set(d2r(t.rotation[0]), d2r(t.rotation[1]), d2r(t.rotation[2]), "XYZ");
    p.scale.fromArray(t.scale.map((s) => Math.max(0.01, s)));
    p.updateMatrixWorld(true);
    this.gizmoHelper.updateMatrixWorld(true);
    this.updateSelectionBox();
  }

  resetTransform(group: string): Transform | null {
    const p = this.pivots.get(group), h = this.home.get(group);
    if (!p || !h) return null;
    p.position.copy(h.position); p.quaternion.copy(h.quaternion); p.scale.copy(h.scale);
    this.updateSelectionBox();
    return this.getTransform(group);
  }

  isMovable(group: string | null): boolean { return !!group && this.pivots.has(group); }

  async exportEdited(onlyVisible: boolean): Promise<Blob> {
    if (!this.root) throw new Error("No scene loaded");
    const restore = new Map<THREE.Mesh, THREE.Material | THREE.Material[]>();
    for (const e of this.entries.values()) { restore.set(e.mesh, e.mesh.material); e.mesh.material = e.original; }
    try {
      const result = await new GLTFExporter().parseAsync(this.root, { binary: true, onlyVisible });
      return new Blob([result as ArrayBuffer], { type: "model/gltf-binary" });
    } finally {
      restore.forEach((mat, mesh) => { mesh.material = mat; });
    }
  }

  screenshot(): string { return this.renderer.domElement.toDataURL("image/png"); }

  stats(): { meshes: number; triangles: number; visible: number } {
    let triangles = 0, visible = 0;
    for (const e of this.entries.values()) {
      const idx = e.mesh.geometry.index;
      const tri = (idx ? idx.count : e.mesh.geometry.attributes.position.count) / 3;
      triangles += tri;
      if (e.mesh.visible) visible++;
    }
    return { meshes: this.entries.size, triangles: Math.round(triangles), visible };
  }

  dispose(): void {
    cancelAnimationFrame(this.frame);
    this.resizeObs.disconnect();
    this.clear();
    this.gizmo.detach();
    this.gizmo.dispose();
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this.axisCanvas.remove();
  }

  // ---------------------------------------------------------------- internals
  static keyOf(meta: SceneNode): string { return meta.group || meta.node; }
  static matches(meta: SceneNode, key: string): boolean { return SceneViewer.keyOf(meta) === key; }

  private selectedMeshes(): THREE.Mesh[] {
    const key = this.selectedGroup;
    return key ? [...this.entries.values()].filter((e) => SceneViewer.matches(e.meta, key)).map((e) => e.mesh) : [];
  }

  private attachGizmo(): void {
    const pivot = this.selectedGroup ? this.pivots.get(this.selectedGroup) : undefined;
    if (pivot && this.tool !== "select") { this.gizmo.attach(pivot); this.gizmo.setMode(this.tool); }
    else this.gizmo.detach();
  }

  private keyCenter(key: string): number[] {
    const box = new THREE.Box3();
    let has = false;
    for (const e of this.entries.values()) if (SceneViewer.matches(e.meta, key)) {
      if (e.meta.bounds) {
        const [min, max] = e.meta.bounds;
        box.expandByPoint(new THREE.Vector3(...(min as [number, number, number])));
        box.expandByPoint(new THREE.Vector3(...(max as [number, number, number])));
        has = true;
      } else {
        box.expandByObject(e.mesh);
        has = true;
      }
    }
    return has ? box.getCenter(new THREE.Vector3()).toArray() : [0, 0, 0];
  }

  private clear(): void {
    this.gizmo.detach();
    if (this.root) {
      this.scene.remove(this.root);
      this.root.traverse((o) => {
        const m = o as THREE.Mesh;
        if (m.isMesh) { m.geometry.dispose(); (Array.isArray(m.material) ? m.material : [m.material]).forEach((mat) => mat.dispose()); }
      });
    }
    this.root = null;
    this.entries.clear();
    this.pivots.clear();
    this.home.clear();
    this.select(null);
  }

  private applyShading(): void {
    for (const e of this.entries.values()) {
      const orig = (Array.isArray(e.original) ? e.original[0] : e.original) as THREE.MeshStandardMaterial;
      if (e.mesh.material !== e.original) (e.mesh.material as THREE.Material).dispose();
      orig.wireframe = this.shading === "lit" && this.wireframe;
      let mat: THREE.Material;
      switch (this.shading) {
        case "lit": mat = e.original as THREE.Material; break;
        case "clay": mat = new THREE.MeshStandardMaterial({ color: 0xb9bcbe, roughness: 0.9, side: THREE.DoubleSide, wireframe: this.wireframe }); break;
        case "truth": mat = new THREE.MeshLambertMaterial({ color: STATUS_COLORS[e.meta.status], side: THREE.DoubleSide, wireframe: this.wireframe }); break;
        case "check": mat = checkMaterial(STATUS_COLORS[e.meta.status], this.wireframe); break;
        default: mat = new THREE.MeshBasicMaterial({ map: orig.map ?? null, color: orig.map ? 0xffffff : orig.color, side: THREE.DoubleSide, wireframe: this.wireframe, vertexColors: !!e.mesh.geometry.attributes.color });
      }
      e.mesh.material = mat;
    }
  }

  private pick(e: MouseEvent | PointerEvent): void {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, this.camera);
    const targets = [...this.entries.values()].filter((x) => x.mesh.visible).map((x) => x.mesh);
    const hit = ray.intersectObjects(targets, false)[0];
    this.select(hit ? (hit.object.userData.key as string) : null);
    this.onSelect(this.selectedGroup);
  }

  private updateSelectionBox(): void {
    if (this.selectionBox) { this.scene.remove(this.selectionBox); this.selectionBox.dispose(); this.selectionBox = null; }
    const meshes = this.selectedMeshes().filter((m) => m.visible);
    if (!meshes.length) return;
    const box = new THREE.Box3();
    meshes.forEach((m) => { m.updateMatrixWorld(true); box.expandByObject(m); });
    this.selectionBox = new THREE.Box3Helper(box, new THREE.Color(0xf7a520));
    this.scene.add(this.selectionBox);
  }

  /** Small X/Y/Z indicator in the corner, like a desktop editor's view axis. */
  private drawAxes(): void {
    const ctx = this.axisCanvas.getContext("2d");
    if (!ctx) return;
    const s = this.axisCanvas.width, c = s / 2, len = s * 0.3;
    ctx.clearRect(0, 0, s, s);
    const q = this.camera.quaternion.clone().invert();
    const axes = [
      { v: new THREE.Vector3(1, 0, 0), color: "#e5484d", label: "X" },
      { v: new THREE.Vector3(0, 1, 0), color: "#46a758", label: "Y" },
      { v: new THREE.Vector3(0, 0, 1), color: "#3e8ef7", label: "Z" },
    ].map((a) => ({ ...a, p: a.v.applyQuaternion(q) })).sort((a, b) => a.p.z - b.p.z);
    ctx.lineWidth = 3;
    ctx.font = "600 18px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const a of axes) {
      const x = c + a.p.x * len, y = c - a.p.y * len;
      ctx.strokeStyle = a.color;
      ctx.beginPath(); ctx.moveTo(c, c); ctx.lineTo(x, y); ctx.stroke();
      ctx.fillStyle = a.color;
      ctx.fillText(a.label, c + a.p.x * (len + 14), c - a.p.y * (len + 14));
    }
  }

  private syncOrtho(): void {
    const rect = this.container.getBoundingClientRect();
    const aspect = rect.width / Math.max(rect.height, 1);
    const dist = this.camera.position.distanceTo(this.controls.target) || 3;
    const half = dist * Math.tan(d2r(this.persp.fov / 2));
    Object.assign(this.ortho, { left: -half * aspect, right: half * aspect, top: half, bottom: -half });
    this.ortho.updateProjectionMatrix();
  }

  private resize(): void {
    const { width, height } = this.container.getBoundingClientRect();
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.persp.aspect = width / height;
    this.persp.updateProjectionMatrix();
    this.syncOrtho();
  }
}
