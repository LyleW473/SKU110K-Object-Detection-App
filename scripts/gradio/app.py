import set_path
import gradio as gr
from ultralytics import YOLO

from src.gradio_app.functions import ImageBatchProcessor, preview_multiple
from src.gradio_app.utils import export_to_csv, cleanup_temp_files

if __name__ == "__main__":
    # Load the model
    model = YOLO(f"final_models/yolo_custom/best.pt")
    processor = ImageBatchProcessor(model=model)

    # Create the Gradio interface with Monochrome theme
    with gr.Blocks(
        theme=gr.themes.Monochrome(primary_hue="pink", secondary_hue="blue"),
        title="Object Detection with YOLO",
        css="""
            .fixed-height-table {
                height: 400px !important;
                position: relative !important;
            }
            .fixed-height-table > div:nth-child(2) {
                max-height: 400px !important;
                overflow-y: auto !important;
            }
            .fixed-height-table table {
                width: 100% !important;
                border-collapse: separate !important;
                border-spacing: 0 !important;
            }
            .fixed-height-table thead {
                position: sticky !important;
                top: 0 !important;
                z-index: 2 !important;
                background: var(--background-fill-primary) !important;
            }
            .fixed-height-table th {
                background: var(--background-fill-primary) !important;
                border-bottom: 2px solid var(--border-color-primary) !important;
                padding: 8px !important;
                color: var(--body-text-color) !important;
            }
            .fixed-height-table td {
                padding: 8px !important;
            }
            /* Add gallery scroll styles */
            .gallery-scroll {
                overflow-y: auto !important;
                max-height: 500px !important;
            }
            .gallery-scroll > div {
                height: auto !important;
            }
            """
    ) as iface:
        gr.Markdown("# Object Detection with YOLO")
        gr.Markdown("Upload an image to detect objects using YOLOv8. Adjust controls to see live preview.")
        
        with gr.Row():
            # Input column
            with gr.Column(scale=1):
                input_image = gr.File(
                    file_count="multiple",
                    label="Input Images",
                    file_types=["image"]
                )
                
                # Advanced controls
                with gr.Accordion("Advanced Controls", open=True):
                    blur = gr.Slider(
                        minimum=0, maximum=10, value=0, step=0.5,
                        label="Blur Amount",
                        info="Adjust image blur (0 = no blur)"
                    )
                    brightness = gr.Slider(
                        minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                        label="Brightness",
                        info="Adjust image brightness (1 = original)"
                    )
                    rotation = gr.Slider(
                        minimum=-180, maximum=180, value=0, step=5,
                        label="Rotation Angle",
                        info="Rotate image (degrees)"
                    )
                    confidence = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.25, step=0.01,
                        label="Confidence Threshold",
                        info="Adjust detection sensitivity"
                    )
                
                detect_btn = gr.Button("Detect Objects", variant="primary")

            with gr.Column(scale=2) as output_column:
                # Preview container
                with gr.Row(visible=True) as preview_container:
                    preview_image = gr.Gallery(
                        label="Live Preview",
                        show_label=True,
                        elem_id="gallery",
                        columns=2,
                        height=500,
                        allow_preview=True,
                        object_fit="contain",
                        elem_classes=["gallery-scroll"]
                    )
                
                # Result container (initially hidden)
                with gr.Row(visible=False) as result_container:
                    with gr.Column(scale=1):
                        final_output = gr.Gallery(
                            label="Results with Detections",
                            show_label=True,
                            elem_id="result_gallery",
                            columns=2,
                            height=500,
                            allow_preview=True,
                            object_fit="contain",
                            elem_classes=["gallery-scroll"]
                        )
                        with gr.Accordion("Detection Details", open=False):
                            detection_table = gr.Dataframe(
                                headers=["Image", "Confidence", "Top-Left", "Bottom-Right"],
                                wrap=True,
                                value=[],
                                interactive=False,
                                elem_classes=["fixed-height-table"]
                            )
                            with gr.Column():
                                export_btn = gr.Button("Export Results", variant="secondary")
                                download_file = gr.File(
                                    label="Download Results",
                                    show_label=False
                                )
        
        # Connect all preview events
        for component in [input_image, blur, brightness, rotation]:
            component.change(
                fn=preview_multiple,
                inputs=[input_image, blur, brightness, rotation],
                outputs=preview_image
            )

        #Clear preview container on new image uploads
        def on_new_image_upload(files, blur, brightness, rotation):
            # Get new previews
            previews = preview_multiple(files, blur, brightness, rotation)
            # Reset results container and table
            return [
                previews,                  
                gr.Row(visible=True),      
                gr.Row(visible=False),     
                None,                      
                None                       
            ]

        # Update the event connections
        input_image.change(
            fn=on_new_image_upload,
            inputs=[input_image, blur, brightness, rotation],
            outputs=[
                preview_image,
                preview_container,
                result_container,
                detection_table,
                download_file
            ]
        )

        # Component change events for live preview
        for component in [blur, brightness, rotation]:
            component.change(
                fn=preview_multiple,
                inputs=[input_image, blur, brightness, rotation],
                outputs=preview_image
            )

        # Connect the components
        def on_detect_click(*args):
            # Show "Processing" notification
            gr.Info("Detection in progress...")
            # Clear previous results
            yield [None, gr.Row(visible=False), gr.Row(visible=False), None]
            
            # Run detection
            results, data = processor.process_multiple_images(*args)
            
            # Show completion notification
            gr.Info("Detection complete!")
            # Show new results
            yield [results, gr.Row(visible=False), gr.Row(visible=True), data]

        # Update the click handler to use streaming outputs
        detect_btn.click(
            fn=on_detect_click,
            inputs=[input_image, confidence, blur, brightness, rotation],
            outputs=[
                final_output,
                preview_container,
                result_container,
                detection_table
            ],
            queue=True  # Enable queuing for streaming
        )

        # Add export button click handler with cleanup
        def on_export_click(data):
            cleanup_temp_files()
            return export_to_csv(data)

        export_btn.click(
            fn=on_export_click,
            inputs=[detection_table],
            outputs=[download_file]
        )

    iface.launch()
