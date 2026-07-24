import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { createXRStore, XR, XRButton, useXR } from "@react-three/xr";
import { LookingGlassConfig } from "@lookingglass/webxr";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";

const xrStore = createXRStore();

// Debug toggle: ?lkgref=1 renders a reference cube + axes at the origin so we
// can tell framing/session issues apart from point-rendering issues in XR.
const LKG_REF = new URLSearchParams(window.location.search).get("lkgref") === "1";

// Debug bisect: ?lkgt=bg,orbit,rig,points includes only the listed parts
// (plus a reference cube). No lkgt = normal full scene.
const LKG_T = new Set(
  (new URLSearchParams(window.location.search).get("lkgt") || "")
    .split(",")
    .filter(Boolean),
);
const LKG_DEBUG = LKG_T.size > 0;

// sRGB byte -> linear LUT. Three.js treats vertex colors as linear-space and
// converts to sRGB on output; the .splat bytes are already sRGB, so feeding
// them in raw double-applies gamma and washes the colors out.
const SRGB_TO_LINEAR = new Float32Array(256);
for (let i = 0; i < 256; i++) {
  const c = i / 255;
  SRGB_TO_LINEAR[i] =
    c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

// Parses the 32-byte-per-splat format produced by TripoSplat into plain
// arrays. Axis mapping (verified visually against the generated cutout):
// TripoSplat's _DEFAULT_TRANSFORM is [[1,0,0],[0,0,-1],[0,1,0]] applied as
// xyz @ T.T → saved = (x, -z, y) (vendor/triposplat/triposplat.py:124-151).
// In the saved frame the model's up is +Z and its front is +X, so the remap
// to three.js (up +Y, front +Z toward the viewer) is the cyclic permute
//   three = (saved.z, -saved.y, saved.x)   — a (z, x, y) permute with sign
// flips on the last two, no extra yaw needed.
interface SplatData {
  count: number;
  positions: Float32Array; // three.js coords: up +Y, front +Z
  colors: Float32Array; // linear space
  rots: Float32Array;
}

function parseSplat(buffer: ArrayBuffer): SplatData {
  const dv = new DataView(buffer);
  const count = Math.floor(dv.byteLength / 32);
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const scales = new Float32Array(count * 3);
  const opacities = new Float32Array(count);
  const rots = new Float32Array(count * 4);

  for (let i = 0; i < count; i++) {
    const off = i * 32;
    const x = dv.getFloat32(off + 0, true);
    const y = dv.getFloat32(off + 4, true);
    const z = dv.getFloat32(off + 8, true);

    positions[i * 3 + 0] = z;
    positions[i * 3 + 1] = -y;
    positions[i * 3 + 2] = x;

    scales[i * 3 + 0] = dv.getFloat32(off + 12, true);
    scales[i * 3 + 1] = dv.getFloat32(off + 16, true);
    scales[i * 3 + 2] = dv.getFloat32(off + 20, true);

    colors[i * 3 + 0] = SRGB_TO_LINEAR[dv.getUint8(off + 24)];
    colors[i * 3 + 1] = SRGB_TO_LINEAR[dv.getUint8(off + 25)];
    colors[i * 3 + 2] = SRGB_TO_LINEAR[dv.getUint8(off + 26)];
    opacities[i] = dv.getUint8(off + 27) / 255;

    rots[i * 4 + 0] = dv.getUint8(off + 28) / 128 - 1;
    rots[i * 4 + 1] = dv.getUint8(off + 29) / 128 - 1;
    rots[i * 4 + 2] = dv.getUint8(off + 30) / 128 - 1;
    rots[i * 4 + 3] = dv.getUint8(off + 31) / 128 - 1;
  }

  // Old viewer behavior: per-splat size had no visual effect (PointsMaterial
  // ignores the attribute), so sizes are only used implicitly via opacity in
  // the morph fade — keep parsing minimal and match that look.
  void scales;
  void opacities;

  return { count, positions, colors, rots };
}

const MORPH_SECONDS = 1.6;

const SPLAT_VERTEX = /* glsl */ `
  attribute vec3 aPosB;
  attribute vec3 aColA;
  attribute vec3 aColB;
  attribute vec4 aRotA;
  attribute vec4 aRotB;
  attribute float aFadeA;
  attribute float aFadeB;
  uniform float uMorphT;
  uniform float uScale;
  varying vec3 vColor;
  varying float vFade;

  vec4 slerp(vec4 a, vec4 b, float t) {
    float d = dot(a, b);
    vec4 bb = b;
    if (d < 0.0) { bb = -b; d = -d; }
    if (d > 0.9995) return normalize(mix(a, bb, t));
    float th = acos(clamp(d, -1.0, 1.0));
    return (sin((1.0 - t) * th) * a + sin(t * th) * bb) / sin(th);
  }

  void main() {
    vec3 pos = mix(position, aPosB, uMorphT);          // lerp position
    vColor = mix(aColA, aColB, uMorphT);               // lerp color
    vFade = mix(aFadeA, aFadeB, uMorphT);              // grow/shrink fade
    vec4 rot = slerp(aRotA, aRotB, uMorphT);           // slerp rotation
    rot = normalize(rot); // (rotation is not visualized by point sprites)
    vec4 mv = modelViewMatrix * vec4(pos, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = 0.02 * vFade * (uScale / -mv.z);
  }
`;

const SPLAT_FRAGMENT = /* glsl */ `
  uniform float uOpacity;
  varying vec3 vColor;
  varying float vFade;
  void main() {
    if (vFade <= 0.001) discard;
    gl_FragColor = vec4(vColor, uOpacity);
  }
`;

type MorphUniforms = {
  uMorphT: { value: number };
  uScale: { value: number };
  uOpacity: { value: number };
};

function SplatPoints({ url }: { url: string }) {
  const get = useThree((s) => s.get);
  const [group, setGroup] = useState<THREE.Group | null>(null);
  const currentRef = useRef<SplatData | null>(null);
  const morphRef = useRef({ active: false, t: 1 });
  const uniformsRef = useRef<MorphUniforms | null>(null);
  const objectRef = useRef<THREE.Points | null>(null);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      const res = await fetch(url);
      if (!res.ok) throw new Error("splat fetch failed");
      const buffer = await res.arrayBuffer();
      if (cancelled) return;

      const next = parseSplat(buffer);
      const prev = currentRef.current;
      const n = Math.max(prev?.count ?? 0, next.count);

      // A-side = previously committed model, B-side = the new model. Where
      // one side has no point (count mismatch), it shares the position but
      // fades from/to zero size instead of flying in from the origin.
      const aPos = new Float32Array(n * 3);
      const bPos = new Float32Array(n * 3);
      const aCol = new Float32Array(n * 3);
      const bCol = new Float32Array(n * 3);
      const aRot = new Float32Array(n * 4);
      const bRot = new Float32Array(n * 4);
      const aFade = new Float32Array(n);
      const bFade = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const inA = prev !== null && i < prev.count;
        const inB = i < next.count;
        const pi = Math.min(i, (prev?.count ?? next.count) - 1);
        const qi = Math.min(i, next.count - 1);
        aPos.set(prev ? prev.positions.subarray(pi * 3, pi * 3 + 3) : next.positions.subarray(qi * 3, qi * 3 + 3), i * 3);
        bPos.set(next.positions.subarray(qi * 3, qi * 3 + 3), i * 3);
        aCol.set(prev ? prev.colors.subarray(pi * 3, pi * 3 + 3) : next.colors.subarray(qi * 3, qi * 3 + 3), i * 3);
        bCol.set(next.colors.subarray(qi * 3, qi * 3 + 3), i * 3);
        aRot.set(prev ? prev.rots.subarray(pi * 4, pi * 4 + 4) : next.rots.subarray(qi * 4, qi * 4 + 4), i * 4);
        bRot.set(next.rots.subarray(qi * 4, qi * 4 + 4), i * 4);
        aFade[i] = inA ? 1 : 0;
        bFade[i] = inB ? 1 : 0;
      }

      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(aPos, 3));
      geo.setAttribute("aPosB", new THREE.BufferAttribute(bPos, 3));
      geo.setAttribute("aColA", new THREE.BufferAttribute(aCol, 3));
      geo.setAttribute("aColB", new THREE.BufferAttribute(bCol, 3));
      geo.setAttribute("aRotA", new THREE.BufferAttribute(aRot, 4));
      geo.setAttribute("aRotB", new THREE.BufferAttribute(bRot, 4));
      geo.setAttribute("aFadeA", new THREE.BufferAttribute(aFade, 1));
      geo.setAttribute("aFadeB", new THREE.BufferAttribute(bFade, 1));

      const uniforms: MorphUniforms = {
        uMorphT: { value: prev ? 0 : 1 },
        uScale: { value: 1 },
        uOpacity: { value: 0.9 },
      };
      const mat = new THREE.ShaderMaterial({
        vertexShader: SPLAT_VERTEX,
        fragmentShader: SPLAT_FRAGMENT,
        uniforms,
        transparent: true,
      });
      const points = new THREE.Points(geo, mat);
      points.frustumCulled = false; // bounds change every frame while morphing

      // Normalize on the new model's bounds (orientation is already handled
      // by the axis remap in parseSplat).
      const box = new THREE.Box3().setFromBufferAttribute(
        new THREE.BufferAttribute(next.positions, 3),
      );
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3()).length() || 1;
      const s = 2.4 / size;
      const g = new THREE.Group();
      g.scale.setScalar(s);
      g.position.set(-center.x * s, -center.y * s, -center.z * s);
      g.add(points);

      // Fit the flat-view camera only for the very first model. Later loads
      // (morph to a newly generated model) must not yank the camera while
      // the user is inspecting the model.
      if (!prev) {
        const { camera, controls } = get();
        const orbit = controls as OrbitControlsImpl | null;
        if (orbit) orbit.target.set(0, 0, 0);
        camera.position.set(0, 0, 2.4);
        camera.lookAt(0, 0, 0);
        orbit?.update();
      }

      // Swap in the new object and dispose the old one.
      setGroup((old) => {
        if (old) {
          const p = old.children[0] as THREE.Points | undefined;
          p?.geometry.dispose();
          (p?.material as THREE.Material | undefined)?.dispose();
        }
        return g;
      });
      objectRef.current = points;
      uniformsRef.current = uniforms;
      morphRef.current = prev ? { active: true, t: 0 } : { active: false, t: 1 };
      currentRef.current = next;
    })().catch((err) => console.error("splat preview error", err));

    return () => {
      cancelled = true;
    };
  }, [url, get]);

  // Drive the morph animation and keep the point-size scale uniform in sync
  // with the drawing buffer (mirrors PointsMaterial size attenuation).
  useFrame((_, dt) => {
    const u = uniformsRef.current;
    if (!u) return;
    u.uScale.value = get().gl.domElement.height * 0.5;
    const m = morphRef.current;
    if (m.active) {
      m.t = Math.min(1, m.t + dt / MORPH_SECONDS);
      u.uMorphT.value = m.t * m.t * (3 - 2 * m.t); // smoothstep ease-in-out
      if (m.t >= 1) m.active = false;
    }
  });

  // Dispose on unmount.
  useEffect(
    () => () => {
      const p = objectRef.current;
      p?.geometry.dispose();
      (p?.material as THREE.Material | undefined)?.dispose();
      objectRef.current = null;
      uniformsRef.current = null;
      currentRef.current = null;
    },
    [],
  );

  return group ? <primitive object={group} /> : null;
}

// three's WebXRManager overwrites the default camera's transform AND
// projection with the XR (quilt) camera while presenting. On session end the
// flat camera keeps the narrow off-center quilt projection, which makes the
// preview a tiny off-frame blob. Restore the flat projection/pose here.
function RestoreCameraOnExit() {
  const gl = useThree((s) => s.gl);
  const camera = useThree((s) => s.camera);
  const size = useThree((s) => s.size);

  useEffect(() => {
    const onEnd = () => {
      const cam = camera as THREE.PerspectiveCamera;
      cam.fov = 60;
      cam.near = 0.01;
      cam.far = 100;
      cam.aspect = size.width / Math.max(1, size.height);
      cam.zoom = 1;
      cam.updateProjectionMatrix();
      cam.position.set(0, 0, 2.4);
      cam.lookAt(0, 0, 0);
    };
    gl.xr.addEventListener("sessionend", onEnd);
    return () => gl.xr.removeEventListener("sessionend", onEnd);
  }, [gl, camera, size]);

  return null;
}

// While presenting on the Looking Glass, the XR ArrayCamera takes over
// rendering and anything that writes the default camera each frame (drei
// OrbitControls, custom rigs) corrupts the quilt views — that was the black
// screen. The correct interaction surface is LookingGlassConfig itself: the
// polyfill recomputes the quilt poses from it live. WASD orbits via
// trackballX/Z, ArrowUp/Down zooms via targetDiam.
function LKGKeyboard() {
  const gl = useThree((s) => s.gl);
  const keys = useRef<Set<string>>(new Set());

  useEffect(() => {
    const isFormTarget = (e: KeyboardEvent) =>
      e.target instanceof HTMLElement &&
      /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName);
    const down = (e: KeyboardEvent) => {
      if (isFormTarget(e)) return;
      if (gl.xr.isPresenting && e.code.startsWith("Arrow")) e.preventDefault();
      keys.current.add(e.code);
    };
    const up = (e: KeyboardEvent) => keys.current.delete(e.code);
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, [gl]);

  useFrame((_, dt) => {
    if (!gl.xr.isPresenting) return;
    const cfg = LookingGlassConfig;
    const k = keys.current;
    const rotSpeed = 1.0; // rad/s
    const zoomSpeed = 1.5; // targetDiam units/s
    if (k.has("KeyA")) cfg.trackballX -= rotSpeed * dt;
    if (k.has("KeyD")) cfg.trackballX += rotSpeed * dt;
    // Clamp the tilt away from the poles so the view ring up-vector never
    // degenerates (gimbal lock). Note: the polyfill's tilt property is
    // trackballY (there is no trackballZ in @lookingglass/webxr 0.6).
    if (k.has("KeyW"))
      cfg.trackballY = Math.max(-1.2, cfg.trackballY - rotSpeed * dt);
    if (k.has("KeyS"))
      cfg.trackballY = Math.min(1.2, cfg.trackballY + rotSpeed * dt);
    if (k.has("ArrowUp"))
      cfg.targetDiam = Math.max(0.5, cfg.targetDiam - zoomSpeed * dt);
    if (k.has("ArrowDown"))
      cfg.targetDiam = Math.min(10, cfg.targetDiam + zoomSpeed * dt);
    // Keep the azimuth bounded so long sessions don't accumulate float error.
    cfg.trackballX = THREE.MathUtils.euclideanModulo(
      cfg.trackballX + Math.PI,
      2 * Math.PI,
    ) - Math.PI;
  });

  return null;
}

// OrbitControls for the flat preview only. While an XR session is active the
// controls must not run: drei calls controls.update() every frame, which
// writes the (XR-owned) default camera and corrupts the quilt views.
function FlatControls() {
  const session = useXR((s) => s.session);
  return (
    <OrbitControls
      makeDefault
      enabled={!session}
      enableDamping
      dampingFactor={0.05}
      minDistance={0.1}
      maxDistance={50}
    />
  );
}

interface SplatViewportProps {
  splatUrl: string | null;
}

export default function SplatViewport({ splatUrl }: SplatViewportProps) {
  return (
    <div className="splat-viewport">
      <Canvas
        flat // NoToneMapping: splat vertex colors are already final sRGB;
             // R3F's default ACES tone mapping would shift them like lighting.
        gl={{ antialias: true }}
        camera={{ fov: 60, near: 0.01, far: 100, position: [0, 0, 3] }}
        onCreated={({ gl }) => {
          gl.xr.enabled = true;
        }}
      >
        {(!LKG_DEBUG || LKG_T.has("bg")) && (
          <color attach="background" args={["#0d1017"]} />
        )}
        <XR store={xrStore}>
          <RestoreCameraOnExit />
          {(!LKG_DEBUG || LKG_T.has("rig")) && <LKGKeyboard />}
          {(LKG_REF || LKG_DEBUG) && (
            <>
              <mesh>
                <boxGeometry args={[0.5, 0.5, 0.5]} />
                <meshBasicMaterial color="red" />
              </mesh>
              <axesHelper args={[1.5]} />
            </>
          )}
          {(!LKG_DEBUG || LKG_T.has("orbit")) && <FlatControls />}
          {(!LKG_DEBUG || LKG_T.has("points")) && splatUrl && (
            <SplatPoints url={splatUrl} />
          )}
        </XR>
      </Canvas>
      <div className="xr-button-wrap">
        <XRButton store={xrStore} mode="immersive-vr">
          Enter VR
        </XRButton>
      </div>
      {!splatUrl && (
        <div className="viewport-placeholder">No splat loaded</div>
      )}
    </div>
  );
}
