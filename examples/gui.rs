use ash::vk;
use eframe::egui::{self, Color32, FontId, Pos2, Rect, Stroke, Vec2};
use image::GrayImage;
use slangtag::{
    ComputeCommandContext, ComputeDevice, Size,
    detect::{
        AprilTagSettings, BlobPairFilterSettings, DecodeSettings, DetectedTag, DetectionSettings,
        Detector, QuadFitSettings,
    },
    gpu::GpuBuffer,
};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Clone, Copy, PartialEq, Eq)]
enum DecimateMode {
    None,
    X1,
    X2,
    X4,
    X8,
}

impl DecimateMode {
    fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::X1 => "1",
            Self::X2 => "2",
            Self::X4 => "4",
            Self::X8 => "8",
        }
    }

    fn factor(self) -> Option<u8> {
        match self {
            Self::None => None,
            Self::X1 => Some(1),
            Self::X2 => Some(2),
            Self::X4 => Some(4),
            Self::X8 => Some(8),
        }
    }

    fn from_factor(value: Option<u8>) -> Self {
        match value {
            None => Self::None,
            Some(1) => Self::X1,
            Some(2) => Self::X2,
            Some(4) => Self::X4,
            Some(8) => Self::X8,
            _ => Self::X2,
        }
    }
}

#[derive(Clone, Copy)]
struct ViewerSettings {
    decimate: DecimateMode,
    min_white_black_diff: u8,
    min_blob_size: u32,
    min_tag_width: u32,
    tag_width: u32,
    reversed_border: u32,
    normal_border: u32,
    min_cluster_pixels: u32,
    max_cluster_pixels_enabled: bool,
    max_cluster_pixels: u32,
    max_cluster_pixels_perimeter_scale: u32,
    max_nmaxima: u32,
    max_line_fit_mse: f32,
    cos_critical_rad: f32,
    cell_size: u32,
    min_stddev_otsu: f32,
    cell_margin_pixels: u32,
    cell_span: u32,
    detect_inverted_marker: bool,
    max_erroneous_border_bits_rate: f32,
    error_correction_rate: f32,
    max_correction_bits: u32,
}

impl Default for ViewerSettings {
    fn default() -> Self {
        let defaults = DetectionSettings::default();
        Self {
            decimate: DecimateMode::from_factor(defaults.decimate),
            min_white_black_diff: defaults.min_white_black_diff,
            min_blob_size: defaults.min_blob_size,
            min_tag_width: defaults.blob_pair_filter.min_tag_width,
            tag_width: defaults.blob_pair_filter.tag_width,
            reversed_border: defaults.blob_pair_filter.reversed_border,
            normal_border: defaults.blob_pair_filter.normal_border,
            min_cluster_pixels: defaults.blob_pair_filter.min_cluster_pixels,
            max_cluster_pixels_enabled: defaults.blob_pair_filter.max_cluster_pixels.is_some(),
            max_cluster_pixels: defaults.blob_pair_filter.max_cluster_pixels.unwrap_or(0),
            max_cluster_pixels_perimeter_scale: defaults
                .blob_pair_filter
                .max_cluster_pixels_perimeter_scale,
            max_nmaxima: defaults.quad_fit.max_nmaxima,
            max_line_fit_mse: defaults.quad_fit.max_line_fit_mse,
            cos_critical_rad: defaults.quad_fit.cos_critical_rad,
            cell_size: defaults.decode.cell_size,
            min_stddev_otsu: defaults.decode.min_stddev_otsu,
            cell_margin_pixels: defaults.decode.cell_margin_pixels,
            cell_span: defaults.decode.cell_span,
            detect_inverted_marker: defaults.decode.detect_inverted_marker,
            max_erroneous_border_bits_rate: defaults.decode.max_erroneous_border_bits_rate,
            error_correction_rate: defaults.apriltag.error_correction_rate,
            max_correction_bits: defaults.apriltag.max_correction_bits,
        }
    }
}

impl ViewerSettings {
    fn to_detection_settings(self) -> DetectionSettings {
        DetectionSettings {
            decimate: self.decimate.factor(),
            min_white_black_diff: self.min_white_black_diff,
            min_blob_size: self.min_blob_size,
            blob_pair_filter: BlobPairFilterSettings {
                min_tag_width: self.min_tag_width,
                tag_width: self.tag_width,
                reversed_border: self.reversed_border,
                normal_border: self.normal_border,
                min_cluster_pixels: self.min_cluster_pixels,
                max_cluster_pixels: self
                    .max_cluster_pixels_enabled
                    .then_some(self.max_cluster_pixels),
                max_cluster_pixels_perimeter_scale: self.max_cluster_pixels_perimeter_scale,
            },
            quad_fit: QuadFitSettings {
                max_nmaxima: self.max_nmaxima,
                max_line_fit_mse: self.max_line_fit_mse,
                cos_critical_rad: self.cos_critical_rad,
            },
            decode: DecodeSettings {
                cell_size: self.cell_size,
                min_stddev_otsu: self.min_stddev_otsu,
                cell_margin_pixels: self.cell_margin_pixels,
                cell_span: self.cell_span,
                detect_inverted_marker: self.detect_inverted_marker,
                max_erroneous_border_bits_rate: self.max_erroneous_border_bits_rate,
            },
            apriltag: AprilTagSettings {
                error_correction_rate: self.error_correction_rate,
                max_correction_bits: self.max_correction_bits,
            },
        }
    }
}

struct LoadedImage {
    path: PathBuf,
    width: u32,
    height: u32,
    gray: GrayImage,
    uploaded_input: Option<UploadedInput>,
    texture: egui::TextureHandle,
}

struct UploadedInput {
    buffer: GpuBuffer<u8>,
    size: Size,
    decimate: Option<u8>,
}

struct ViewerApp {
    device: ComputeDevice,
    command_context: ComputeCommandContext,
    settings: ViewerSettings,
    detector: Option<Detector>,
    loaded_image: Option<LoadedImage>,
    detections: Vec<DetectedTag>,
    last_runtime: Option<Duration>,
    status: String,
    auto_run: bool,
}

impl ViewerApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let device = ComputeDevice::new_default();
        let command_context = device.create_command_context();
        let settings = ViewerSettings::default();
        Self {
            device,
            command_context,
            settings,
            detector: None,
            loaded_image: None,
            detections: Vec::new(),
            last_runtime: None,
            status: "Load an image to begin.".to_owned(),
            auto_run: true,
        }
    }

    fn rebuild_detector(&mut self) {
        let detection_settings = self.settings.to_detection_settings();
        if let Some(image) = self.loaded_image.as_ref() {
            self.detector = normalized_size_for_decimate(
                image.width,
                image.height,
                detection_settings.decimate,
            )
            .and_then(|size| Detector::new(self.device.clone(), detection_settings, size).ok());
        } else {
            self.detector = None;
        }
    }

    fn load_image(&mut self, ctx: &egui::Context, path: PathBuf) {
        match image::open(&path) {
            Ok(image) => {
                let rgba = image.to_rgba8();
                let [width, height] = [rgba.width(), rgba.height()];
                let color_image = egui::ColorImage::from_rgba_unmultiplied(
                    [width as usize, height as usize],
                    rgba.as_raw(),
                );
                let texture =
                    ctx.load_texture("source-image", color_image, egui::TextureOptions::LINEAR);
                self.loaded_image = Some(LoadedImage {
                    path: path.clone(),
                    width,
                    height,
                    gray: image.to_luma8(),
                    uploaded_input: None,
                    texture,
                });
                self.rebuild_detector();
                self.detections.clear();
                self.last_runtime = None;
                self.status = format!("Loaded {}", path.display());
            }
            Err(error) => {
                self.status = format!("Failed to load {}: {}", path.display(), error);
            }
        }
    }

    fn run_detection(&mut self) {
        let Some(_) = self.loaded_image.as_ref() else {
            self.status = "Load an image first.".to_owned();
            return;
        };
        if self.ensure_uploaded_input().is_err() {
            self.last_runtime = None;
            self.status = "Detection failed. One or more parameter values are invalid.".to_owned();
            self.detections.clear();
            return;
        }
        let Some(image) = self.loaded_image.as_ref() else {
            self.last_runtime = None;
            self.status = "Load an image first.".to_owned();
            self.detections.clear();
            return;
        };
        let Some(uploaded_input) = image.uploaded_input.as_ref() else {
            self.last_runtime = None;
            self.status = "Failed to prepare GPU input image.".to_owned();
            self.detections.clear();
            return;
        };

        let start = std::time::Instant::now();
        let Some(detector) = self.detector.as_ref() else {
            self.last_runtime = None;
            self.status = "Detection failed. Invalid image size or settings.".to_owned();
            self.detections.clear();
            return;
        };
        match detector.detect_descriptor(
            &mut self.command_context,
            uploaded_input.buffer.descriptor(),
            uploaded_input.size,
        ) {
            Ok(output) => {
                self.last_runtime = Some(start.elapsed());
                self.status = format!("Detected {} tags", output.tags.len());
                self.detections = output.tags;
            }
            Err(_) => {
                self.last_runtime = None;
                self.status =
                    "Detection failed. One or more parameter values are invalid.".to_owned();
                self.detections.clear();
            }
        }
    }

    fn ensure_uploaded_input(&mut self) -> Result<(), ()> {
        let decimate = self.settings.decimate.factor();
        let Some(image) = self.loaded_image.as_mut() else {
            return Err(());
        };

        if image
            .uploaded_input
            .as_ref()
            .is_some_and(|uploaded| uploaded.decimate == decimate)
        {
            return Ok(());
        }

        let aligned_gray =
            crop_image_to_multiple(image.gray.clone(), 4 * decimate.unwrap_or(1) as u32)?;
        let uploaded_size = Size::new(aligned_gray.width(), aligned_gray.height());
        let uploaded_buffer = self.device.upload_buffer(
            &mut self.command_context,
            aligned_gray.as_raw(),
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            true,
        );
        image.uploaded_input = Some(UploadedInput {
            buffer: uploaded_buffer,
            size: uploaded_size,
            decimate,
        });
        Ok(())
    }

    fn show_settings(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        ui.horizontal(|ui| {
            ui.label("Decimate");
            egui::ComboBox::from_id_salt("decimate_mode")
                .selected_text(self.settings.decimate.label())
                .show_ui(ui, |ui| {
                    changed |= ui
                        .selectable_value(&mut self.settings.decimate, DecimateMode::None, "None")
                        .changed();
                    changed |= ui
                        .selectable_value(&mut self.settings.decimate, DecimateMode::X1, "1")
                        .changed();
                    changed |= ui
                        .selectable_value(&mut self.settings.decimate, DecimateMode::X2, "2")
                        .changed();
                    changed |= ui
                        .selectable_value(&mut self.settings.decimate, DecimateMode::X4, "4")
                        .changed();
                    changed |= ui
                        .selectable_value(&mut self.settings.decimate, DecimateMode::X8, "8")
                        .changed();
                });
        });

        changed |= ui
            .add(
                egui::Slider::new(&mut self.settings.min_white_black_diff, 0..=255)
                    .text("min_white_black_diff"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.settings.min_blob_size, 1..=1000).text("min_blob_size"),
            )
            .changed();

        ui.separator();
        ui.label("Blob Pair Filter");
        changed |= ui
            .add(egui::Slider::new(&mut self.settings.min_tag_width, 1..=20).text("min_tag_width"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.settings.tag_width, 1..=8).text("tag_width"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.settings.reversed_border, 0..=4)
                    .text("reversed_border"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.settings.normal_border, 0..=4).text("normal_border"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.settings.min_cluster_pixels, 1..=5000)
                    .text("min_cluster_pixels"),
            )
            .changed();
        changed |= ui
            .checkbox(
                &mut self.settings.max_cluster_pixels_enabled,
                "limit max_cluster_pixels",
            )
            .changed();
        if self.settings.max_cluster_pixels_enabled {
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.settings.max_cluster_pixels, 1..=20000)
                        .text("max_cluster_pixels"),
                )
                .changed();
        }
        changed |= ui
            .add(
                egui::Slider::new(
                    &mut self.settings.max_cluster_pixels_perimeter_scale,
                    1..=32,
                )
                .text("max_cluster_pixels_perimeter_scale"),
            )
            .changed();

        ui.separator();
        ui.label("Quad Fit");
        changed |= ui
            .add(egui::Slider::new(&mut self.settings.max_nmaxima, 1..=64).text("max_nmaxima"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.settings.max_line_fit_mse, 0.0..=100.0)
                    .text("max_line_fit_mse"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.settings.cos_critical_rad, 0.0..=1.0)
                    .text("cos_critical_rad"),
            )
            .changed();

        ui.separator();
        ui.label("Decode");
        changed |= ui
            .add(egui::Slider::new(&mut self.settings.cell_size, 1..=16).text("cell_size"))
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.settings.min_stddev_otsu, 0.0..=50.0)
                    .text("min_stddev_otsu"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.settings.cell_margin_pixels, 0..=8)
                    .text("cell_margin_pixels"),
            )
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.settings.cell_span, 1..=8).text("cell_span"))
            .changed();
        changed |= ui
            .checkbox(
                &mut self.settings.detect_inverted_marker,
                "detect_inverted_marker",
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.settings.max_erroneous_border_bits_rate, 0.0..=1.0)
                    .text("max_erroneous_border_bits_rate"),
            )
            .changed();

        ui.separator();
        ui.label("AprilTag");
        changed |= ui
            .add(
                egui::Slider::new(&mut self.settings.error_correction_rate, 0.0..=1.0)
                    .text("error_correction_rate"),
            )
            .changed();
        changed |= ui
            .add(
                egui::Slider::new(&mut self.settings.max_correction_bits, 0..=4)
                    .text("max_correction_bits"),
            )
            .changed();

        changed
    }

    fn draw_detection_overlay(
        painter: &egui::Painter,
        image_rect: Rect,
        image_size: Vec2,
        detections: &[DetectedTag],
    ) {
        for tag in detections {
            let points: [Pos2; 4] = tag.corners.map(|corner| {
                let x = image_rect.left() + (corner[0] / image_size.x) * image_rect.width();
                let y = image_rect.top() + (corner[1] / image_size.y) * image_rect.height();
                Pos2::new(x, y)
            });

            for i in 0..4 {
                painter.line_segment(
                    [points[i], points[(i + 1) % 4]],
                    Stroke::new(2.0, Color32::from_rgb(80, 255, 120)),
                );
            }

            let label = match tag.id {
                Some(id) => format!("id {}", id),
                None => "id ?".to_owned(),
            };
            painter.text(
                points[0],
                egui::Align2::LEFT_TOP,
                label,
                FontId::proportional(14.0),
                Color32::from_rgb(255, 240, 120),
            );
        }
    }
}

fn normalized_size_for_decimate(width: u32, height: u32, decimate: Option<u8>) -> Option<Size> {
    let factor = decimate.unwrap_or(1) as u32;
    let multiple = 4 * factor;
    let normalized_width = width - (width % multiple);
    let normalized_height = height - (height % multiple);
    (normalized_width > 0 && normalized_height > 0)
        .then_some(Size::new(normalized_width, normalized_height))
}

fn crop_image_to_multiple(image: GrayImage, multiple: u32) -> Result<GrayImage, ()> {
    let width = image.width();
    let height = image.height();
    let cropped_width = width - (width % multiple);
    let cropped_height = height - (height % multiple);

    if cropped_width == 0 || cropped_height == 0 {
        return Err(());
    }

    if cropped_width == width && cropped_height == height {
        return Ok(image);
    }

    Ok(image::imageops::crop_imm(&image, 0, 0, cropped_width, cropped_height).to_image())
}

impl eframe::App for ViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut run_requested = false;
        let mut settings_changed = false;

        egui::SidePanel::left("controls")
            .resizable(true)
            .default_width(320.0)
            .show(ctx, |ui| {
                ui.heading("slangtag Viewer");
                if ui.button("Load Image").clicked()
                    && let Some(path) = rfd::FileDialog::new()
                        .add_filter("Image", &["png", "jpg", "jpeg", "webp", "bmp", "tiff"])
                        .pick_file()
                {
                    self.load_image(ctx, path);
                    if self.auto_run {
                        run_requested = true;
                    }
                }

                if ui.button("Run Detection").clicked() {
                    run_requested = true;
                }
                ui.checkbox(&mut self.auto_run, "Auto-run on parameter changes");

                if let Some(image) = &self.loaded_image {
                    ui.label(format!("Image: {}", image.path.display()));
                    ui.label(format!("Resolution: {}x{}", image.width, image.height));
                } else {
                    ui.label("No image loaded");
                }

                if let Some(duration) = self.last_runtime {
                    ui.label(format!("Runtime: {:.2?}", duration));
                }
                ui.label(format!("Detections: {}", self.detections.len()));
                ui.separator();
                ui.label(&self.status);
                ui.separator();

                settings_changed = self.show_settings(ui);
            });

        if settings_changed {
            self.rebuild_detector();
        }
        if run_requested || (self.auto_run && settings_changed) {
            self.run_detection();
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(image) = &self.loaded_image {
                let available = ui.available_size();
                let image_size = Vec2::new(image.width as f32, image.height as f32);
                let scale = (available.x / image_size.x)
                    .min(available.y / image_size.y)
                    .clamp(0.05, 10.0);
                let draw_size = image_size * scale;

                let (rect, _) = ui.allocate_exact_size(draw_size, egui::Sense::hover());
                ui.painter().image(
                    image.texture.id(),
                    rect,
                    Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
                Self::draw_detection_overlay(ui.painter(), rect, image_size, &self.detections);
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("Load an image to preview detections.");
                });
            }
        });
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("slangtag Viewer")
            .with_inner_size([1400.0, 900.0]),
        ..Default::default()
    };
    eframe::run_native(
        "slangtag Viewer",
        options,
        Box::new(|cc| Ok(Box::new(ViewerApp::new(cc)))),
    )
}
