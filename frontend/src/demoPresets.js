/**
 * GeoSense AI Preset Scenarios
 * Corresponds to Section 34 & 40 of SIH 2026 Master Prompt.
 */

export const DEMO_PRESETS = [
  {
    id: 'captioning-grounding',
    title: 'Scene Captioning & Grounding',
    category: 'Single Image',
    description: 'Generates detailed scene caption with RemoteCLIP & Florence-2 and grounds prominent landmarks.',
    query: 'Provide a detailed description of this scene and detect all visible infrastructure and roads.',
    task: 'captioning',
    suggestedRoles: ['optical'],
    sampleImages: [
      {
        name: 'sample_port_optical.tif',
        role: 'optical',
        preview: 'https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?w=600&auto=format&fit=crop&q=80',
        meta: { gsd: '0.5m', crs: 'EPSG:32643', bands: 3, format: 'GTiff' }
      }
    ]
  },
  {
    id: 'vqa-counting',
    title: 'Visual Q&A (Count & Presence)',
    category: 'Targeted Query',
    description: 'Precision object detection and count aggregation powered by Grounding-DINO and SmolVLM.',
    query: 'How many fuel storage tanks and cargo ships are present in the harbor?',
    task: 'vqa',
    suggestedRoles: ['optical'],
    sampleImages: [
      {
        name: 'sample_industrial_complex.tif',
        role: 'optical',
        preview: 'https://images.unsplash.com/photo-1578328819058-b69f3a3b0f6b?w=600&auto=format&fit=crop&q=80',
        meta: { gsd: '0.3m', crs: 'EPSG:32643', bands: 3, format: 'GTiff' }
      }
    ]
  },
  {
    id: 'bitemporal-change',
    title: 'Bi-Temporal Change Analysis',
    category: 'Multi-Temporal',
    description: 'ECC affine alignment, radiometric equalization, and Otsu difference thresholding for damage or flood analysis.',
    query: 'What changed between these two dates? Highlight newly submerged or cleared vegetation.',
    task: 'change_analysis',
    suggestedRoles: ['before', 'after'],
    sampleImages: [
      {
        name: 'coastal_t1_before.tif',
        role: 'before',
        preview: 'https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=600&auto=format&fit=crop&q=80',
        meta: { gsd: '10.0m', crs: 'EPSG:32643', bands: 3, format: 'GTiff' }
      },
      {
        name: 'coastal_t2_after.tif',
        role: 'after',
        preview: 'https://images.unsplash.com/photo-1518495973542-4542c06a5843?w=600&auto=format&fit=crop&q=80',
        meta: { gsd: '10.0m', crs: 'EPSG:32643', bands: 3, format: 'GTiff' }
      }
    ]
  },
  {
    id: 'optical-sar-fusion',
    title: 'Optical–SAR Sensor Fusion',
    category: 'Multimodal',
    description: 'Sentinel-1 Lee speckle filtered SAR backscatter combined with cloud-obscured Sentinel-2 optical imagery.',
    query: 'Fuse the cloud-obscured optical image with SAR backscatter to map water bodies and urban structures.',
    task: 'optical_sar_fusion',
    suggestedRoles: ['optical', 'sar'],
    sampleImages: [
      {
        name: 'sentinel2_cloudy_optical.tif',
        role: 'optical',
        preview: 'https://images.unsplash.com/photo-1534088568595-a066f410bcda?w=600&auto=format&fit=crop&q=80',
        meta: { gsd: '10.0m', crs: 'EPSG:32643', bands: 3, format: 'GTiff' }
      },
      {
        name: 'sentinel1_grd_sar.tif',
        role: 'sar',
        preview: 'https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?w=600&auto=format&fit=crop&q=80',
        meta: { gsd: '10.0m', crs: 'EPSG:32643', bands: 1, format: 'GTiff' }
      }
    ]
  }
];

export const QUICK_PROMPTS = [
  "Describe this scene.",
  "Locate all buildings and compute their footprint area.",
  "How many storage tanks are visible?",
  "Where are the water bodies located?",
  "What changed between the before and after acquisitions?",
  "Map water bodies and urban extent using optical and SAR fusion."
];

/**
 * Creates a synthetic satellite test file for instantaneous preset demonstration.
 */
export async function generatePresetImageFile(name, role, seed = 1) {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');

  if (role === 'sar') {
    // Generate radar speckle / grayscale texture
    const imgData = ctx.createImageData(512, 512);
    for (let i = 0; i < imgData.data.length; i += 4) {
      const speckle = Math.floor(Math.random() * 140) + 40;
      imgData.data[i] = speckle;
      imgData.data[i + 1] = speckle;
      imgData.data[i + 2] = speckle;
      imgData.data[i + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);

    // Draw high-backscatter bright urban returns and dark water calm surface
    ctx.fillStyle = '#05070a';
    ctx.beginPath();
    ctx.ellipse(360, 260, 120, 80, Math.PI / 4, 0, 2 * Math.PI);
    ctx.fill();

    ctx.fillStyle = '#ffffff';
    ctx.fillRect(100, 100, 60, 40);
    ctx.fillRect(180, 120, 50, 50);
  } else if (role === 'before') {
    // Green lush vegetation and coast
    ctx.fillStyle = '#1e3a29'; // deep forest
    ctx.fillRect(0, 0, 512, 512);
    ctx.fillStyle = '#0f766e'; // river / water
    ctx.beginPath();
    ctx.moveTo(0, 200);
    ctx.bezierCurveTo(180, 250, 320, 150, 512, 280);
    ctx.lineWidth = 45;
    ctx.strokeStyle = '#0284c7';
    ctx.stroke();
  } else if (role === 'after') {
    // Submerged or deforested after event
    ctx.fillStyle = '#3f3120'; // cleared / mud
    ctx.fillRect(0, 0, 512, 512);
    ctx.beginPath();
    ctx.moveTo(0, 190);
    ctx.bezierCurveTo(180, 260, 320, 140, 512, 290);
    ctx.lineWidth = 140; // flooded extent!
    ctx.strokeStyle = '#0369a1';
    ctx.stroke();
  } else {
    // Optical RGB (urban, roads, buildings, water)
    ctx.fillStyle = '#334155'; // urban slate base
    ctx.fillRect(0, 0, 512, 512);

    // Roads
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 14;
    ctx.beginPath();
    ctx.moveTo(0, 256);
    ctx.lineTo(512, 256);
    ctx.moveTo(256, 0);
    ctx.lineTo(256, 512);
    ctx.stroke();

    // Vegetation park
    ctx.fillStyle = '#166534';
    ctx.beginPath();
    ctx.roundRect(40, 40, 160, 160, 12);
    ctx.fill();

    // Water reservoir
    ctx.fillStyle = '#0284c7';
    ctx.beginPath();
    ctx.ellipse(380, 380, 90, 70, 0, 0, 2 * Math.PI);
    ctx.fill();

    // Industrial tanks / circular features
    ctx.fillStyle = '#e2e8f0';
    ctx.beginPath();
    ctx.arc(380, 120, 28, 0, 2 * Math.PI);
    ctx.arc(450, 120, 28, 0, 2 * Math.PI);
    ctx.arc(380, 185, 28, 0, 2 * Math.PI);
    ctx.fill();

    // Buildings
    ctx.fillStyle = '#94a3b8';
    ctx.fillRect(60, 300, 70, 60);
    ctx.fillRect(150, 310, 80, 60);
    ctx.fillRect(60, 390, 90, 70);
  }

  return new Promise((resolve) => {
    canvas.toBlob((blob) => {
      const file = new File([blob], name, { type: 'image/png' });
      resolve({
        file,
        name,
        size: (file.size / (1024 * 1024)).toFixed(2) + ' MB',
        preview: URL.createObjectURL(blob),
        role,
        isPreset: true,
        meta: { gsd: '0.5m', crs: 'EPSG:32643', bands: role === 'sar' ? 1 : 3, format: 'PNG/GTiff' }
      });
    }, 'image/png');
  });
}
