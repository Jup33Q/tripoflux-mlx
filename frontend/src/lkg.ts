// Looking Glass (LKG) WebXR adaptation, ported from Three/Demo01.
// Must be imported before any WebGLRenderer is constructed (see main.tsx).
import {
  LookingGlassWebXRPolyfill,
  LookingGlassConfig,
} from "@lookingglass/webxr";

const config = LookingGlassConfig;
config.targetY = 0;
config.targetZ = 0;
config.targetDiam = 3;
config.fovy = (14 * Math.PI) / 180;
new LookingGlassWebXRPolyfill();

// Local debug without a device: ?lkgdebug=quilt shows the raw quilt grid on
// the page canvas while presenting; ?lkgdebug=center shows the center view.
// (InlineView: Swizzled = 0, Center = 1, Quilt = 2)
const lkgDebug = new URLSearchParams(window.location.search).get("lkgdebug");
if (lkgDebug === "quilt") config.inlineView = 2;
else if (lkgDebug === "center") config.inlineView = 1;

// The Looking Glass polyfill provides a polyfilled XRSession, but Three.js
// r173+ detects native XRWebGLBinding and tries to use WebXR Layers. Native
// XRWebGLBinding rejects a polyfilled session, so force the legacy XRWebGLLayer
// path (which the polyfill does implement) by hiding the binding class while
// the WebGLRenderer/WebXRManager is constructed.
// https://github.com/mrdoob/three.js/issues/31432#issuecomment-4429081922
// @ts-expect-error - intentionally deleting the native binding
delete window.XRWebGLBinding;
