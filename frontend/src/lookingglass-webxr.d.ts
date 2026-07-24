// @lookingglass/webxr ships no TypeScript declarations; declare the surface
// this project uses.
declare module '@lookingglass/webxr' {
  export interface LookingGlassConfigShape {
    quiltResolution: { width: number; height: number };
    columns: number;
    rows: number;
    targetX: number;
    targetY: number;
    targetZ: number;
    trackballX: number;
    trackballY: number;
    targetDiam: number;
    fovy: number;
    depthiness: number;
    inlineView: number;
  }

  export const LookingGlassConfig: LookingGlassConfigShape;

  export class LookingGlassWebXRPolyfill {
    constructor();
  }
}
