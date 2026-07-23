// Gaussian Splat viewer using Three.js with OrbitControls (ES modules).
// Parses the 32-byte-per-splat format produced by TripoSplat.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

class SplatViewer {
  constructor(canvas, buffer) {
    this.canvas = canvas;
    this.buffer = buffer;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0d1017);

    const aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
    this.camera = new THREE.PerspectiveCamera(60, aspect, 0.01, 100);
    this.camera.position.set(0, 0, 3);

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    this.renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.minDistance = 0.1;
    this.controls.maxDistance = 50;
    this.controls.enablePan = true;
    this.controls.enableZoom = true;
    this.controls.enableRotate = true;

    this._parse();
    this._build();
    this._bindResize();
  }

  _parse() {
    const dv = new DataView(this.buffer);
    const count = dv.byteLength / 32;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const scales = new Float32Array(count * 3);
    const rotations = new Float32Array(count * 4);
    const opacities = new Float32Array(count);

    for (let i = 0; i < count; i++) {
      const off = i * 32;
      const x = dv.getFloat32(off + 0, true);
      const y = dv.getFloat32(off + 4, true);
      const z = dv.getFloat32(off + 8, true);

      // TripoSplat applies _DEFAULT_TRANSFORM before saving:
      //   new_x = old_x, new_y = old_z, new_z = -old_y
      // Three.js uses right-handed: X right, Y up, Z toward viewer.
      // We map splat (x, y, z) -> three.js (x, z, -y) so the model appears upright.
      positions[i * 3 + 0] = x;
      positions[i * 3 + 1] = z;
      positions[i * 3 + 2] = -y;

      scales[i * 3 + 0] = dv.getFloat32(off + 12, true);
      scales[i * 3 + 1] = dv.getFloat32(off + 16, true);
      scales[i * 3 + 2] = dv.getFloat32(off + 20, true);

      colors[i * 3 + 0] = dv.getUint8(off + 24) / 255;
      colors[i * 3 + 1] = dv.getUint8(off + 25) / 255;
      colors[i * 3 + 2] = dv.getUint8(off + 26) / 255;
      opacities[i] = dv.getUint8(off + 27) / 255;

      rotations[i * 4 + 0] = dv.getUint8(off + 28) / 128 - 1;
      rotations[i * 4 + 1] = dv.getUint8(off + 29) / 128 - 1;
      rotations[i * 4 + 2] = dv.getUint8(off + 30) / 128 - 1;
      rotations[i * 4 + 3] = dv.getUint8(off + 31) / 128 - 1;
    }

    this.positions = positions;
    this.colors = colors;
    this.scales = scales;
    this.rotations = rotations;
    this.opacities = opacities;
  }

  _build() {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(this.colors, 3));

    // Use per-splat opacity to modulate point size as a cheap approximation.
    const sizes = new Float32Array(this.positions.length / 3);
    for (let i = 0; i < sizes.length; i++) {
      const sx = this.scales[i * 3 + 0];
      const sy = this.scales[i * 3 + 1];
      const sz = this.scales[i * 3 + 2];
      sizes[i] = Math.cbrt(sx * sy * sz) * 0.5 * this.opacities[i];
    }
    geo.setAttribute("size", new THREE.BufferAttribute(sizes, 1));

    const mat = new THREE.PointsMaterial({
      size: 0.02,
      vertexColors: true,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.9,
    });

    this.points = new THREE.Points(geo, mat);
    this.scene.add(this.points);

    // Fit camera to splat bounds.
    const box = new THREE.Box3().setFromBufferAttribute(geo.getAttribute("position"));
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length();
    this.controls.target.copy(center);
    this.camera.position.copy(center).add(new THREE.Vector3(0, 0, size * 0.8));
    this.controls.update();
  }

  _bindResize() {
    window.addEventListener("resize", () => {
      const w = this.canvas.clientWidth;
      const h = this.canvas.clientHeight;
      this.camera.aspect = w / Math.max(1, h);
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h, false);
    });
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame(() => this.render());
  }

  dispose() {
    if (this.points) {
      this.points.geometry.dispose();
      this.points.material.dispose();
    }
    this.renderer.dispose();
  }
}

export default SplatViewer;
