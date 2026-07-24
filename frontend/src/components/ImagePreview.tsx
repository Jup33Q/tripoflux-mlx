interface DownloadLinks {
  ply: string;
  spz: string;
  splat: string;
}

interface ImagePreviewProps {
  fluxUrl: string | null;
  rgbaUrl: string | null;
  downloads: DownloadLinks | null;
  jobId: string | null;
}

export default function ImagePreview({
  fluxUrl,
  rgbaUrl,
  downloads,
  jobId,
}: ImagePreviewProps) {
  return (
    <>
      <div className="preview-card">
        <h3>Generated Image</h3>
        {fluxUrl ? (
          <img src={fluxUrl} alt="Generated" />
        ) : (
          <div className="preview-placeholder">No image yet</div>
        )}
      </div>
      <div className="preview-card">
        <h3>Background Removed</h3>
        <div className="checkerboard">
          {rgbaUrl ? (
            <img src={rgbaUrl} alt="RGBA" />
          ) : (
            <div className="preview-placeholder">No image yet</div>
          )}
        </div>
      </div>
      {downloads && jobId && (
        <div className="preview-card download-card">
          <h3>Downloads</h3>
          <div className="links">
            <a href={downloads.ply} download={`${jobId}.ply`}>
              Download .ply
            </a>
            <a href={downloads.spz} download={`${jobId}.spz`}>
              Download .spz
            </a>
            <a href={downloads.splat} download={`${jobId}.splat`}>
              Download .splat
            </a>
          </div>
        </div>
      )}
    </>
  );
}
