import os
import string
from pathlib import Path

import gradio as gr
import pandas as pd

from src.director_tools import colorize, declutter, emotion, line_art, remove_bg, sketch
from src.generate_images import main as generate_images
from src.upscale_images import anime4k, realcugan_ncnn_vulkan, waifu2x_caffe
from utils import (
    copy_current_img,
    del_current_img,
    load_plugins,
    move_current_img,
    plugin_list,
    read_json,
    remove_pnginfo,
    restart,
    show_first_img,
    show_next_img,
    stop_generate,
    tagger,
    tk_asksavefile_asy,
)
from utils.components import (
    add_character,
    add_precise_reference,
    add_wildcard,
    add_wildcard_to_textbox,
    auto_complete,
    del_precise_reference,
    delete_character,
    delete_wildcard,
    enable_plugin,
    install_plugin,
    modify_wildcard,
    return_character_reference_component_visible,
    return_image2image_visible,
    return_inpaint_input_image_mode,
    return_pnginfo,
    return_position_interactive,
    send_pnginfo_to_generate,
    uninstall_plugin,
    update_components_for_models_change,
    update_components_for_sampler_change,
    update_components_for_sm_change,
    update_from_dropdown,
    update_from_height,
    update_from_width,
    update_repo,
    update_wildcard_names,
    update_wildcard_tags,
)
from utils.environment import env
from utils.image_tools import return_array_image
from utils.prepare import _model, is_updated, last_data, parameters
from utils.setting_updater import modify_env
from utils.variable import (
    BASE_PATH,
    CHARACTER_POSITION,
    CR_MODE,
    MODELS,
    NOISE_SCHEDULE,
    RESOLUTION,
    SAMPLER,
    UC_PRESET,
    WILDCARD_TYPE,
)

with gr.Blocks(
    theme=env.theme if env.theme != "無" else None,
    title="Auto-NovelAI-Refactor",
) as anr:
    announcement = gr.Row()
    with announcement:
        with gr.Column(scale=2):
            updata_warning = gr.Markdown(
                '<span style="color: green; font-size: 20px;">新增塗鴉重繪功能！</span>',
                show_label=False,
            )
        user_read = gr.Checkbox(label="我已知悉", interactive=True, scale=1)
        gr.HTML("")
        user_read.change(
            lambda: gr.update(visible=False),
            inputs=None,
            outputs=announcement,
        )
    with gr.Row():
        model = gr.Dropdown(
            choices=MODELS,
            value=_model,
            label="生圖模型",
            interactive=True,
            scale=1,
        )
        with gr.Column(scale=2):
            gr.Markdown(
                "# [Auto-NovelAI-Refactor](https://github.com/zhulinyv/Auto-NovelAI-Refactor) | NovelAI 批次產生工具 | 版本: "
                + is_updated
            )

    with gr.Row():
        with gr.Column(scale=3):
            positive_input = gr.TextArea(
                value=last_data.get("input"),
                label="正面提示詞",
                placeholder="請在此輸入正面提示詞...",
                lines=5,
            )
            auto_complete(positive_input)
            negative_input = gr.TextArea(
                value=parameters.get("negative_prompt"),
                label="負面提示詞",
                placeholder="請在此輸入負面提示詞...",
                lines=5,
            )
            auto_complete(negative_input)
        with gr.Column(scale=1):
            with gr.Row():
                furry_mode = gr.Button(
                    "🌸", visible=False if _model in ["nai-diffusion-3", "nai-diffusion-furry-3"] else True
                )
                furry_mode.click(lambda x: "🐾" if x == "🌸" else "🌸", inputs=furry_mode, outputs=furry_mode)
                add_quality_tags = gr.Checkbox(
                    value=parameters.get("qualityToggle", True), label="新增品質詞", interactive=True
                )
            undesired_contentc_preset = gr.Dropdown(
                choices=[
                    x
                    for x in UC_PRESET
                    if x
                    not in {
                        "nai-diffusion-4-5-full": [],
                        "nai-diffusion-4-5-curated": ["Furry Focus"],
                        "nai-diffusion-4-full": ["Furry Focus", "Human Focus"],
                        "nai-diffusion-4-curated-preview": ["Furry Focus", "Human Focus"],
                        "nai-diffusion-3": ["Furry Focus"],
                        "nai-diffusion-furry-3": ["Furry Focus", "Human Focus"],
                    }.get(_model, [])
                ],
                value="None" if parameters.get("negative_prompt") else "Heavy",
                label="負面提示詞預設",
                interactive=True,
            )
            generate_button = gr.Button(value="開始產生")
            stop_button = gr.Button(value="停止產生")
            stop_button.click(stop_generate)
            quantity = gr.Slider(
                minimum=1,
                maximum=999,
                value=1,
                step=1,
                label="產生數量",
                interactive=True,
            )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tab(label="參數設定"):
                resolution = gr.Dropdown(
                    choices=RESOLUTION + ["自訂"],
                    value=(
                        "自訂"
                        if (res := "{}x{}".format(parameters.get("width"), parameters.get("height"))) not in RESOLUTION
                        else res
                    ),
                    label="解析度預設",
                    interactive=True,
                )
                with gr.Row():
                    width = gr.Slider(
                        minimum=0,
                        maximum=50000,
                        value=parameters.get("width", 832),
                        step=64,
                        label="寬",
                        interactive=True,
                    )
                    height = gr.Slider(
                        minimum=0,
                        maximum=50000,
                        value=parameters.get("height", 1216),
                        step=64,
                        label="高",
                        interactive=True,
                    )
                resolution.change(
                    fn=update_from_dropdown,
                    inputs=[resolution],
                    outputs=[width, height],
                )
                width.change(
                    fn=update_from_width,
                    inputs=[width, height, resolution],
                    outputs=resolution,
                )
                height.change(
                    fn=update_from_height,
                    inputs=[width, height, resolution],
                    outputs=resolution,
                )
                steps = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=parameters.get("steps", 23),
                    label="取樣步數",
                    step=1,
                    interactive=True,
                )
                prompt_guidance = gr.Slider(
                    minimum=0,
                    maximum=10,
                    value=parameters.get("scale", 5),
                    label="提示詞引導係數",
                    step=0.1,
                    interactive=True,
                )
                prompt_guidance_rescale = gr.Slider(
                    minimum=0,
                    maximum=10,
                    value=parameters.get("cfg_rescale", 0),
                    label="提示詞重取樣係數",
                    step=0.02,
                    interactive=True,
                )
                with gr.Row():
                    variety = gr.Checkbox(
                        value=True if parameters.get("skip_cfg_above_sigma") else False,
                        label="Variety+",
                        interactive=True,
                    )
                    decrisp = gr.Checkbox(
                        value=parameters.get("dynamic_thresholding", False),
                        label="Decrisp",
                        visible=True if _model in ["nai-diffusion-3", "nai-diffusion-furry-3"] else False,
                        interactive=True,
                    )
                with gr.Row():
                    sm = gr.Checkbox(
                        value=parameters.get("sm", False),
                        label="SMEA",
                        visible=True if _model in ["nai-diffusion-3", "nai-diffusion-furry-3"] else False,
                        interactive=True,
                    )
                    sm_dyn = gr.Checkbox(
                        value=parameters.get("sm_dyn", False),
                        label="DYN",
                        visible=True if _model in ["nai-diffusion-3", "nai-diffusion-furry-3"] else False,
                        interactive=True,
                    )
                with gr.Row():
                    seed = gr.Textbox(value="-1", label="種子", interactive=True, scale=4)
                with gr.Row(scale=1):
                    last_seed = gr.Button(value="♻️", size="sm")
                    random_seed = gr.Button(value="🎲", size="sm")
                    last_seed.click(
                        lambda: read_json("last.json")["parameters"]["seed"] if os.path.exists("last.json") else "-1",
                        outputs=seed,
                    )
                    random_seed.click(lambda: "-1", outputs=seed)
                sampler = gr.Dropdown(
                    choices=(
                        SAMPLER
                        if _model in ["nai-diffusion-3", "nai-diffusion-furry-3"]
                        else [x for x in SAMPLER if x != "ddim_v3"]
                    ),
                    value=parameters.get("sampler", "k_euler_ancestral"),
                    label="取樣器",
                    interactive=True,
                )
                noise_schedule = gr.Dropdown(
                    choices=(
                        NOISE_SCHEDULE
                        if _model in ["nai-diffusion-3", "nai-diffusion-furry-3"]
                        else [x for x in NOISE_SCHEDULE if x != "native"]
                    ),
                    value=parameters.get("noise_schedule", "karras"),
                    label="調度器",
                    interactive=True,
                )
                legacy_uc = gr.Checkbox(
                    value=parameters.get("legacy_uc", False),
                    label="Legacy Prompt Conditioning Mode",
                    visible=(True if _model in ["nai-diffusion-4-full", "nai-diffusion-4-curated-preview"] else False),
                    interactive=True,
                )
                with gr.Column():
                    inpaint_input_image_mode = gr.Radio(
                        ["圖生圖", "局部重繪", "塗鴉重繪"],
                        value="圖生圖",
                        show_label=False,
                        visible=False,
                        interactive=True,
                    )
                    inpaint_input_image = gr.ImageEditor(
                        width=650,
                        height=650,
                        sources=["upload", "clipboard", "webcam"],
                        brush=False,
                        eraser=False,
                        type="pil",
                        label="基礎圖片（可選）",
                        layers=False,
                    )
                inpaint_i2i_strength = gr.Slider(
                    0.01, 1, 1, step=0.01, label="Mask Strength", visible=False, interactive=True
                )
                strength = gr.Slider(0.01, 0.99, 0.7, step=0.01, label="強度", visible=False, interactive=True)
                noise = gr.Slider(0, 10, 0, step=0.01, label="噪點", visible=False, interactive=True)
                inpaint_input_image.change(
                    return_image2image_visible,
                    inputs=[inpaint_input_image, inpaint_input_image_mode],
                    outputs=[
                        inpaint_input_image,
                        strength,
                        noise,
                        width,
                        height,
                        inpaint_input_image_mode,
                        inpaint_i2i_strength,
                    ],
                )
                inpaint_input_image_mode.change(
                    return_inpaint_input_image_mode,
                    [inpaint_input_image_mode, inpaint_input_image],
                    [inpaint_input_image, inpaint_i2i_strength],
                )
            character_position_tab = gr.Tab(
                label="角色分區", visible=False if _model in ["nai-diffusion-3", "nai-diffusion-furry-3"] else True
            )
            with character_position_tab:
                character_components_list = []
                character_components_number = gr.Number(value=0, visible=False)  # 使用 Number 替代 Slider
                with gr.Row():
                    add_character_button = gr.Button("新增角色")
                    delete_character_button = gr.Button("刪除角色")
                character_position_table = gr.Dataframe(
                    value=pd.DataFrame(
                        [
                            ["1", "A1", "B1", "C1", "D1", "E1"],
                            ["2", "A2", "B2", "C2", "D2", "E2"],
                            ["3", "A3", "B3", "C3", "D3", "E3"],
                            ["4", "A4", "B4", "C4", "D4", "E4"],
                            ["5", "A5", "B5", "C5", "D5", "E5"],
                        ],
                        columns=["位置", "A", "B", "C", "D", "E"],
                    ),
                    visible=False,
                    interactive=False,
                )
                ai_choice = gr.Checkbox(True, label="AI's Choice (Character Positions (Global))", interactive=False)
                ai_choice.change(lambda x: gr.update(visible=not x), inputs=ai_choice, outputs=character_position_table)
                gr.Markdown("<hr>")

                # 先建立所有元件
                for i in range(6):
                    character_components_list.append(
                        gr.TextArea(label=f"角色 {i+1} 正面提示詞", lines=3, visible=False, interactive=True)
                    )
                    character_components_list.append(
                        gr.TextArea(label=f"角色 {i+1} 負面提示詞", lines=3, visible=False, interactive=True)
                    )
                    with gr.Row():
                        character_components_list.append(
                            gr.Dropdown(
                                choices=CHARACTER_POSITION,
                                label=f"角色 {i+1} 位置",
                                visible=False,
                                interactive=True,
                            )
                        )
                        character_components_list.append(
                            gr.Checkbox(False, label="啟用", visible=False, interactive=True)
                        )
                    character_components_list.append(gr.Markdown("<hr>", visible=False))

                add_character_button.click(
                    add_character,
                    inputs=character_components_number,
                    outputs=[ai_choice, character_components_number] + character_components_list,
                )
                delete_character_button.click(
                    delete_character,
                    inputs=character_components_number,
                    outputs=[ai_choice, character_components_number] + character_components_list,
                )
                ai_choice.change(return_position_interactive, inputs=ai_choice, outputs=character_components_list)
            character_reference_tab = gr.Tab(
                "角色參考",
                visible=True if _model in ["nai-diffusion-4-5-full", "nai-diffusion-4-5-curated"] else False,
            )
            with character_reference_tab:
                precise_reference_components_list = []
                precise_reference_components_number = gr.Number(value=0, visible=False)
                with gr.Row():
                    precise_reference_add_btn = gr.Button("新增角色")
                    precise_reference_del_btn = gr.Button("刪除角色")
                gr.Markdown("<hr>")
                gr.Markdown(
                    "新增角色並啟用時，每張圖片消耗 5 點數；由於 Gradio 動態渲染限制，ANR 無法無限新增角色參考圖，目前上限為 10 張，如需新增更多請加群回饋"
                )
                for i in range(10):
                    with gr.Row():
                        precise_reference_components_list.append(
                            gr.Image(type="filepath", show_label=False, visible=False, interactive=True)
                        )
                        with gr.Column():
                            with gr.Row():
                                precise_reference_components_list.append(
                                    gr.Checkbox(False, label="啟用", visible=False, interactive=True)
                                )
                                precise_reference_components_list.append(
                                    gr.Dropdown(
                                        CR_MODE,
                                        value="character&style",
                                        show_label=False,
                                        visible=False,
                                        interactive=True,
                                    )
                                )
                            precise_reference_components_list.append(
                                gr.Slider(0, 1, 1, step=0.05, label="Strength", visible=False, interactive=True)
                            )
                            precise_reference_components_list.append(
                                gr.Slider(0, 1, 1, step=0.05, label="Fidelity", visible=False, interactive=True)
                            )
                    precise_reference_components_list.append(gr.Markdown("<hr>", visible=False))

            vibe_transfer_tab = gr.Tab(label="風格遷移", visible=True, interactive=True)
            precise_reference_add_btn.click(
                add_precise_reference,
                inputs=precise_reference_components_number,
                outputs=[vibe_transfer_tab, precise_reference_components_number] + precise_reference_components_list,
            )
            precise_reference_del_btn.click(
                del_precise_reference,
                inputs=precise_reference_components_number,
                outputs=[vibe_transfer_tab, precise_reference_components_number] + precise_reference_components_list,
            )

            with vibe_transfer_tab:
                naiv4vibebundle_file = gr.File(
                    type="filepath",
                    label="*.naiv4vibebundle",
                    visible=True if _model not in ["nai-diffusion-3", "nai-diffusion-furry-3"] else False,
                    interactive=True,
                )
                naiv4vibebundle_file.change(
                    return_character_reference_component_visible,
                    inputs=[model, naiv4vibebundle_file],
                    outputs=character_reference_tab,
                )
                normalize_reference_strength_multiple = gr.Checkbox(
                    True,
                    label="Normalize Reference Strength Values",
                    visible=True if _model not in ["nai-diffusion-3", "nai-diffusion-furry-3"] else False,
                    interactive=True,
                )
                naiv4vibebundle_file_instruction = gr.Markdown(
                    "關於 *.naiv4vibebundle 檔案的取得：請先在官網上傳 vibe 使用的底圖，調整權重後進行編碼，待全部圖片完成編碼後下載 *.naiv4vibebundle 檔案，注意不要下載單張圖片編碼的 vibe 檔案",
                    visible=True if _model not in ["nai-diffusion-3", "nai-diffusion-furry-3"] else False,
                )
                nai3vibe_column = gr.Column(
                    visible=True if _model in ["nai-diffusion-3", "nai-diffusion-furry-3"] else False
                )
                with nai3vibe_column:
                    nai3vibe_transfer_image_count = gr.State(1)
                    nai3vibe_transfer_add_button = gr.Button("新增圖片")
                    nai3vibe_transfer_del_button = gr.Button("刪除圖片")
                    nai3vibe_transfer_add_button.click(
                        lambda x: x + 1,
                        nai3vibe_transfer_image_count,
                        nai3vibe_transfer_image_count,
                    )
                    nai3vibe_transfer_del_button.click(
                        lambda x: x - 1 if x >= 1 else x,
                        nai3vibe_transfer_image_count,
                        nai3vibe_transfer_image_count,
                    )
                    gr.Markdown("<hr>")

                    @gr.render(inputs=nai3vibe_transfer_image_count)
                    def _(count):
                        nai3vibe_transfer_components_list = []
                        for _ in range(count):
                            with gr.Row():
                                nai3vibe_transfer_image = gr.Image(type="filepath")
                                with gr.Column():
                                    reference_information_extracted_multiple = gr.Slider(
                                        0, 1, 1.0, step=0.01, label="資訊擷取強度", interactive=True
                                    )
                                    reference_strength_multiple = gr.Slider(
                                        0, 1, 0.6, step=0.01, label="畫風參考強度", interactive=True
                                    )
                                    gr.Markdown("<hr>")
                            nai3vibe_transfer_components_list.append(nai3vibe_transfer_image)
                            nai3vibe_transfer_components_list.append(reference_information_extracted_multiple)
                            nai3vibe_transfer_components_list.append(reference_strength_multiple)
                        generate_button.click(
                            generate_images,
                            inputs=[
                                model,
                                positive_input,
                                negative_input,
                                furry_mode,
                                add_quality_tags,
                                undesired_contentc_preset,
                                quantity,
                                width,
                                height,
                                steps,
                                prompt_guidance,
                                prompt_guidance_rescale,
                                variety,
                                seed,
                                sampler,
                                noise_schedule,
                                decrisp,
                                sm,
                                sm_dyn,
                                legacy_uc,
                                inpaint_input_image,
                                inpaint_input_image_mode,
                                inpaint_i2i_strength,
                                strength,
                                noise,
                                naiv4vibebundle_file,
                                normalize_reference_strength_multiple,
                                ai_choice,
                            ]
                            + character_components_list
                            + precise_reference_components_list
                            + nai3vibe_transfer_components_list,
                            outputs=[output_image, output_information],
                        )

            with gr.Tab(label="Wildcards"):
                with gr.Tab("使用或修改"):
                    wildcard_type = gr.Dropdown(
                        choices=WILDCARD_TYPE,
                        value=None,
                        label="分類",
                        interactive=True,
                    )
                    wildcard_name = gr.Dropdown(
                        value=None,
                        label="名稱",
                        interactive=True,
                    )
                    wildcard_tags = gr.Textbox(label="包含的提示詞", lines=2, interactive=True)
                    with gr.Row():
                        wildcard_add_positive = gr.Button("新增到正面提示詞")
                        wildcard_add_negative = gr.Button("新增到負面提示詞")
                    with gr.Row():
                        wildcard_modify = gr.Button("修改", size="sm")
                        wildcard_delete = gr.Button("刪除", size="sm")
                with gr.Tab("建立新卡片"):
                    with gr.Row():
                        select_new_wildcard_type = gr.Dropdown(
                            choices=WILDCARD_TYPE, value=None, label="從已有分類中選擇", interactive=True
                        )
                        new_wildcard_type = gr.Textbox(label="分類")
                        select_new_wildcard_type.change(lambda x: x, select_new_wildcard_type, new_wildcard_type)
                    new_wildcard_name = gr.Textbox(label="名稱")
                    new_wildcard_tags = gr.Textbox(label="提示詞", lines=2)
                    wildcard_add = gr.Button("新增卡片")
                wildcard_refresh = gr.Button("重新整理列表")

                wildcard_type.change(update_wildcard_names, inputs=wildcard_type, outputs=wildcard_name)
                wildcard_name.change(
                    update_wildcard_tags,
                    inputs=[wildcard_type, wildcard_name],
                    outputs=wildcard_tags,
                )
                wildcard_add_positive.click(
                    add_wildcard_to_textbox,
                    inputs=[positive_input, wildcard_type, wildcard_name],
                    outputs=positive_input,
                )
                wildcard_add_negative.click(
                    add_wildcard_to_textbox,
                    inputs=[negative_input, wildcard_type, wildcard_name],
                    outputs=negative_input,
                )
                wildcard_refresh.click(
                    lambda: (
                        gr.update(choices=os.listdir("./wildcards")),
                        gr.update(choices=os.listdir("./wildcards")),
                    ),
                    outputs=[wildcard_type, select_new_wildcard_type],
                )
        with gr.Column(scale=2):
            with gr.Tab("圖片產生"):
                with gr.Column(scale=2):
                    output_image = gr.Gallery(label="輸出圖片", interactive=False, show_label=False)
                    output_information = gr.Textbox(label="輸出資訊", interactive=False, show_label=False)
                    wildcard_modify.click(
                        modify_wildcard,
                        inputs=[wildcard_type, wildcard_name, wildcard_tags],
                        outputs=output_information,
                    )
                    wildcard_delete.click(
                        delete_wildcard,
                        inputs=[wildcard_type, wildcard_name],
                        outputs=output_information,
                    )
                    wildcard_add.click(
                        add_wildcard,
                        inputs=[new_wildcard_type, new_wildcard_name, new_wildcard_tags],
                        outputs=output_information,
                    )
            with gr.Tab("導演工具"):
                director_input_path = gr.Textbox(label="批次處理路徑（同時輸入路徑和圖片時僅處理圖片）")
                with gr.Row():
                    director_input_image = gr.Image(type="filepath", label="Input")
                    director_output_image = gr.Gallery(interactive=False, label="Output")
                with gr.Tab("Remove BG"):
                    remove_bg_button = gr.Button("開始處理")
                    remove_bg_button.click(
                        remove_bg, inputs=[director_input_path, director_input_image], outputs=director_output_image
                    )
                with gr.Tab("Line Art"):
                    line_art_button = gr.Button("開始處理")
                    line_art_button.click(
                        line_art, inputs=[director_input_path, director_input_image], outputs=director_output_image
                    )
                with gr.Tab("Sketch"):
                    sketch_button = gr.Button("開始處理")
                    sketch_button.click(
                        sketch, inputs=[director_input_path, director_input_image], outputs=director_output_image
                    )
                with gr.Tab("Colorize"):
                    with gr.Row():
                        colorize_defry = gr.Slider(0, 5, 0, step=1, label="Defry")
                        colorize_prompt = gr.Textbox(label="Prompt (Optional)")
                    colorize_button = gr.Button("開始處理")
                    colorize_button.click(
                        colorize,
                        inputs=[director_input_path, director_input_image, colorize_defry, colorize_prompt],
                        outputs=director_output_image,
                    )
                with gr.Tab("Emotion"):
                    with gr.Row():
                        emotion_tag = gr.Dropdown(
                            [
                                "Neutral",
                                "Happy",
                                "Sad",
                                "Angry",
                                "Scared",
                                "Surprised",
                                "Tired",
                                "Excited",
                                "Nervous",
                                "Thinking",
                                "Confused",
                                "Shy",
                                "Disgusted",
                                "Smug",
                                "Bored",
                                "Laughing",
                                "Irritated",
                                "Aroused",
                                "Embarrassed",
                                "Worried",
                                "Love",
                                "Determined",
                                "Hurt",
                                "Playful",
                            ],
                            value="Neutral",
                            label="Emotion",
                            interactive=True,
                        )
                        emotion_strength = gr.Dropdown(
                            ["Normal", "Slightly Weak", "Weak", "Even Weaker", "Very Weak", "Weakest"],
                            show_label=False,
                            interactive=True,
                        )
                        emotion_prompt = gr.Textbox(label="Prompt (Optional)")
                    emotion_button = gr.Button("開始處理")
                    emotion_button.click(
                        emotion,
                        inputs=[
                            director_input_path,
                            director_input_image,
                            emotion_tag,
                            emotion_strength,
                            emotion_prompt,
                        ],
                        outputs=director_output_image,
                    )
                with gr.Tab("Declutter"):
                    declutter_button = gr.Button("開始處理")
                    declutter_button.click(
                        declutter, inputs=[director_input_path, director_input_image], outputs=director_output_image
                    )
                director_stop_button = gr.Button("停止處理")
                director_stop_button.click(stop_generate)
            with gr.Tab("超分降噪"):
                upscale_input_path = gr.Textbox(label="批次處理路徑（同時輸入路徑和圖片時僅處理圖片）")
                with gr.Row():
                    with gr.Column():
                        upscale_input_image = gr.Image(type="numpy", interactive=False, label="Input")
                        with gr.Row():
                            upscale_input_text = gr.Textbox(visible=False)
                            upscale_input_btn = gr.Button("選擇圖片")
                            upscale_clear_btn = gr.Button("清除選擇")
                    upscale_clear_btn.click(lambda x: x, gr.Textbox(None, visible=False), upscale_input_text)
                    upscale_input_btn.click(tk_asksavefile_asy, inputs=[], outputs=[upscale_input_text])
                    upscale_input_text.change(return_array_image, upscale_input_text, upscale_input_image)
                    upscale_output_image = gr.Gallery(interactive=False, label="Output")
                with gr.Tab("realcugan-ncnn-vulkan"):
                    with gr.Column():
                        # gr.Markdown("發生錯誤時請確保電腦上有 vulkan-1.dll 檔案")
                        with gr.Row():
                            realcugan_noise = gr.Slider(minimum=-1, maximum=3, value=3, step=1, label="降噪強度")
                            realcugan_scale = gr.Slider(minimum=2, maximum=4, value=2, step=1, label="放大倍數")
                        realcugan_model = gr.Radio(
                            ["models-se", "models-pro", "models-nose"], value="models-se", label="超分模型"
                        )
                        realcugan_button = gr.Button("開始產生")
                        realcugan_button.click(
                            realcugan_ncnn_vulkan,
                            inputs=[
                                upscale_input_path,
                                upscale_input_text,
                                realcugan_noise,
                                realcugan_scale,
                                realcugan_model,
                            ],
                            outputs=upscale_output_image,
                        )
                with gr.Tab("Anime4K"):
                    with gr.Column():
                        # gr.Markdown("發生錯誤時請確保電腦上有 OpenCL.dll 檔案")
                        with gr.Row():
                            anime4k_zoomFactor = gr.Slider(1, maximum=32, value=2, step=1, label="放大倍數")
                            anime4k_HDNLevel = gr.Slider(minimum=1, maximum=3, step=1, value=3, label="HDN 等級")
                        with gr.Row():
                            anime4k_GPUMode = gr.Radio([True, False], label="開啟 GPU 加速", value=True)
                            anime4k_CNNMode = gr.Radio([True, False], label="開啟 ACNet 模式", value=True)
                            anime4k_HDN = gr.Radio([True, False], label="為 ACNet 開啟 HDN", value=True)
                        anime4k_button = gr.Button("開始產生")
                        anime4k_button.click(
                            anime4k,
                            inputs=[
                                upscale_input_path,
                                upscale_input_text,
                                anime4k_zoomFactor,
                                anime4k_HDNLevel,
                                anime4k_GPUMode,
                                anime4k_CNNMode,
                                anime4k_HDN,
                            ],
                            outputs=upscale_output_image,
                        )
                with gr.Tab("waifu2x-caffe"):
                    with gr.Column():
                        with gr.Row():
                            waifu2x_caffe_mode = gr.Radio(
                                ["noise", "scale", "noise_scale"], value="noise_scale", label="模式"
                            )
                            waifu2x_caffe_process = gr.Radio(["cpu", "gpu", "cudnn"], value="gpu", label="處理模式")
                            waifu2x_caffe_tta = gr.Radio([True, False], value=False, label="開啟 tta 模式")
                        with gr.Row():
                            waifu2x_caffe_scale = gr.Slider(minimum=1, maximum=32, value=2, label="放大倍數")
                            waifu2x_caffe_noise = gr.Slider(minimum=0, maximum=3, step=1, value=3, label="降噪強度")
                        waifu2x_caffe_model = gr.Radio(
                            [
                                "anime_style_art_rgb",
                                "anime_style_art",
                                "photo",
                                "upconv_7_anime_style_art_rgb",
                                "upconv_7_photo",
                                "upresnet10",
                                "cunet",
                                "ukbench",
                            ],
                            value="cunet",
                            label="超分模型",
                        )
                        waifu2x_caffe_button = gr.Button("開始產生")
                        waifu2x_caffe_button.click(
                            waifu2x_caffe,
                            inputs=[
                                upscale_input_path,
                                upscale_input_text,
                                waifu2x_caffe_mode,
                                waifu2x_caffe_process,
                                waifu2x_caffe_tta,
                                waifu2x_caffe_scale,
                                waifu2x_caffe_noise,
                                waifu2x_caffe_model,
                            ],
                            outputs=upscale_output_image,
                        )
                upscale_stop_button = gr.Button("停止產生")
                upscale_stop_button.click(stop_generate)
            with gr.Tab("法術解析"):
                with gr.Tab("讀取資訊"):
                    with gr.Row():
                        with gr.Column():
                            pnginfo_image = gr.Image(type="pil", image_mode="RGBA")
                            send_button = gr.Button("發送到圖片產生", visible=False)
                            send_info_from_json = gr.Files(
                                type="filepath",
                                visible=False,
                                interactive=True,
                                label="*.json 文件",
                                file_count="single",
                                file_types=[".json"],
                            )
                            send_info_from_json.change(
                                send_pnginfo_to_generate,
                                inputs=send_info_from_json,
                                outputs=[
                                    positive_input,
                                    negative_input,
                                    width,
                                    height,
                                    steps,
                                    prompt_guidance,
                                    prompt_guidance_rescale,
                                    variety,
                                    decrisp,
                                    sm,
                                    sm_dyn,
                                    seed,
                                    sampler,
                                    noise_schedule,
                                    legacy_uc,
                                    add_quality_tags,
                                    undesired_contentc_preset,
                                    ai_choice,
                                    character_components_number,
                                ]
                                + character_components_list,
                            )
                            with gr.Row():
                                show_all_pnginfo = gr.Checkbox(False, label="顯示所有資訊")
                                show_send_info_from_json = gr.Checkbox(False, label="從 json 檔案匯入")
                                show_send_info_from_json.change(
                                    lambda x: gr.update(visible=True if x else False),
                                    inputs=show_send_info_from_json,
                                    outputs=send_info_from_json,
                                )
                        with gr.Column():
                            source = gr.Textbox(label="Source")
                            generation_time = gr.Textbox(label="Generation time")
                            comment = gr.JSON(label="Comment", open=True)
                            description = gr.TextArea(label="Description")
                            software = gr.Textbox(label="Software")
                    all_pnginfo = gr.JSON(label="全部資訊", open=True, visible=False)
                    show_all_pnginfo.change(
                        lambda x: gr.update(visible=x), inputs=show_all_pnginfo, outputs=all_pnginfo
                    )
                    pnginfo_image.change(
                        return_pnginfo,
                        inputs=pnginfo_image,
                        outputs=[
                            send_button,
                            source,
                            generation_time,
                            comment,
                            description,
                            software,
                            all_pnginfo,
                        ],
                    )
                    send_button.click(
                        send_pnginfo_to_generate,
                        inputs=pnginfo_image,
                        outputs=[
                            positive_input,
                            negative_input,
                            width,
                            height,
                            steps,
                            prompt_guidance,
                            prompt_guidance_rescale,
                            variety,
                            decrisp,
                            sm,
                            sm_dyn,
                            seed,
                            sampler,
                            noise_schedule,
                            legacy_uc,
                            add_quality_tags,
                            undesired_contentc_preset,
                            ai_choice,
                            character_components_number,
                        ]
                        + character_components_list,
                    )
                with gr.Tab("圖片反推"):
                    with gr.Row():
                        with gr.Column():
                            tagger_image = gr.Image(type="filepath", label="Input")
                            tagger_model = gr.Dropdown(
                                choices=[
                                    "SmilingWolf/wd-swinv2-tagger-v3",
                                    "SmilingWolf/wd-convnext-tagger-v3",
                                    "SmilingWolf/wd-vit-tagger-v3",
                                    "SmilingWolf/wd-vit-large-tagger-v3",
                                    "SmilingWolf/wd-eva02-large-tagger-v3",
                                    "SmilingWolf/wd-v1-4-moat-tagger-v2",
                                    "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
                                    "SmilingWolf/wd-v1-4-convnext-tagger-v2",
                                    "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
                                    "SmilingWolf/wd-v1-4-vit-tagger-v2",
                                    "deepghs/idolsankaku-swinv2-tagger-v1",
                                    "deepghs/idolsankaku-eva02-large-tagger-v1",
                                ],
                                value="SmilingWolf/wd-swinv2-tagger-v3",
                                label="Model",
                            )
                            with gr.Row():
                                general_tags_threshold = gr.Slider(
                                    0, 1, 0.35, step=0.05, label="General Tags Threshold"
                                )
                                use_mcut_threshold_general = gr.Checkbox(False, label="Use MCut threshold")
                            with gr.Row():
                                character_tags_threshold = gr.Slider(
                                    0, 1, 0.85, step=0.05, label="Character Tags Threshold"
                                )
                                use_mcut_threshold_character = gr.Checkbox(False, label="Use MCut threshold")
                            with gr.Row():
                                submit_button = gr.Button("提交")
                                tagger_send_button = gr.Button("發送到圖片產生")
                        with gr.Column():
                            tagger_sorted_general_strings = gr.TextArea(label="Output (string)", interactive=False)
                            tagger_rating = gr.Label(label="Rating")
                            tagger_character_res = gr.Label(label="Output (characters)")
                            tagger_general_res = gr.Label(label="Output (tags)")
                        submit_button.click(
                            tagger,
                            inputs=[
                                tagger_image,
                                tagger_model,
                                general_tags_threshold,
                                use_mcut_threshold_general,
                                character_tags_threshold,
                                use_mcut_threshold_character,
                            ],
                            outputs=[
                                tagger_sorted_general_strings,
                                tagger_rating,
                                tagger_character_res,
                                tagger_general_res,
                            ],
                        )
                        tagger_send_button.click(
                            lambda x: x, inputs=tagger_sorted_general_strings, outputs=positive_input
                        )
                with gr.Tab("抹除資料"):
                    with gr.Row():
                        with gr.Column():
                            remove_pnginfo_image = gr.Image(type="numpy", interactive=False, label="單張處理（可選）")
                            with gr.Row():
                                norm_input_text = gr.Textbox(visible=False)
                                norm_input_btn = gr.Button("選擇圖片")
                                norm_clear_btn = gr.Button("清除選擇")
                        norm_clear_btn.click(lambda x: x, gr.Textbox(None, visible=False), norm_input_text)
                        norm_input_btn.click(tk_asksavefile_asy, inputs=[], outputs=[norm_input_text])
                        norm_input_text.change(return_array_image, norm_input_text, remove_pnginfo_image)
                        with gr.Column():
                            remove_pnginfo_generate_button = gr.Button("開始處理")
                            remove_pnginfo_choices = gr.CheckboxGroup(
                                [
                                    "Title",
                                    "Description",
                                    "Software",
                                    "Source",
                                    "Generation time",
                                    "Comment",
                                    "dpi",
                                    "parameters",
                                    "prompt",
                                ],
                                value=[
                                    "Title",
                                    "Description",
                                    "Software",
                                    "Source",
                                    "Generation time",
                                    "Comment",
                                    "dpi",
                                    "parameters",
                                    "prompt",
                                ],
                                label="要清除的內容",
                                scale=2,
                            )
                            remove_pnginfo_metadate = gr.Textbox(label="新增自訂資訊（可選）")
                            remove_pnginfo_input_path = gr.Textbox(label="批次處理路徑（可選）")
                            remove_pnginfo_output_information = gr.Textbox(show_label=False, visible=False)
                            remove_pnginfo_output_information.change(
                                lambda x: gr.update(visible=True if x else False),
                                inputs=remove_pnginfo_output_information,
                                outputs=remove_pnginfo_output_information,
                            )
                            remove_pnginfo_generate_button.click(
                                fn=remove_pnginfo,
                                inputs=[
                                    norm_input_text,
                                    remove_pnginfo_input_path,
                                    remove_pnginfo_choices,
                                    remove_pnginfo_metadate,
                                ],
                                outputs=[remove_pnginfo_output_information],
                            )
            with gr.Tab("圖片篩選"):
                with gr.Column():
                    with gr.Row():
                        selector_input_path = gr.Textbox(label="圖片目錄", scale=4)
                        selector_select_button = gr.Button("載入圖片", scale=1)
                    with gr.Row():
                        selector_output_path = gr.Textbox(label="目錄1")
                        _selector_output_path = gr.Textbox(label="目錄2")
                with gr.Row():
                    with gr.Column(scale=2):
                        selector_output_image = gr.Gallery(preview=True, label="Image")
                        selector_send_image = gr.Button("發送到法術解析")
                    with gr.Column(scale=1):
                        selector_next_button = gr.Button("跳過")
                        with gr.Row():
                            selector_move_button = gr.Button("移動到目錄1", min_width=50)
                            _selector_move_button = gr.Button("移動到目錄2", min_width=50)
                        with gr.Row():
                            selector_copy_button = gr.Button("複製到目錄1", min_width=50)
                            _selector_copy_button = gr.Button("複製到目錄2", min_width=50)
                        selector_delete_button = gr.Button("刪除")
                    selector_current_img = gr.Textbox(visible=False)
                    selector_select_button.click(
                        fn=show_first_img,
                        inputs=[selector_input_path],
                        outputs=[selector_output_image, selector_current_img],
                    )
                    selector_next_button.click(fn=show_next_img, outputs=[selector_output_image, selector_current_img])
                    selector_move_button.click(
                        fn=move_current_img,
                        inputs=[selector_current_img, selector_output_path],
                        outputs=[selector_output_image, selector_current_img],
                    )
                    _selector_move_button.click(
                        fn=move_current_img,
                        inputs=[selector_current_img, _selector_output_path],
                        outputs=[selector_output_image, selector_current_img],
                    )
                    selector_copy_button.click(
                        fn=copy_current_img,
                        inputs=[selector_current_img, selector_output_path],
                        outputs=[selector_output_image, selector_current_img],
                    )
                    _selector_copy_button.click(
                        fn=copy_current_img,
                        inputs=[selector_current_img, _selector_output_path],
                        outputs=[selector_output_image, selector_current_img],
                    )
                    selector_delete_button.click(
                        fn=del_current_img,
                        inputs=[selector_current_img],
                        outputs=[selector_output_image, selector_current_img],
                    )
                    selector_send_image.click(fn=lambda x: x, inputs=selector_current_img, outputs=pnginfo_image)
            with gr.Tab("插件商店"):
                plugin_store_output_information = gr.Textbox(show_label=False, visible=False)
                plugin_store_plugin_name = gr.Dropdown(
                    value=None,
                    choices=list(
                        dict.fromkeys(
                            list(read_json("./assets/plugins.json").keys())
                            + [i.replace(".py", "") for i in os.listdir("./plugins")]
                        )
                    ),
                    label="插件名稱",
                )
                with gr.Row():
                    plugin_store_install_button = gr.Button("安裝／更新")
                    plugin_store_uninstall_button = gr.Button("刪除")
                    plugin_store_enable_button = gr.Button("啟用／停用")
                    plugin_store_restart_button = gr.Button("重啟")
                gr.Markdown(plugin_list())
                plugin_store_install_button.click(
                    install_plugin, inputs=plugin_store_plugin_name, outputs=plugin_store_output_information
                )
                plugin_store_uninstall_button.click(
                    uninstall_plugin, inputs=plugin_store_plugin_name, outputs=plugin_store_output_information
                )
                plugin_store_enable_button.click(
                    enable_plugin, inputs=plugin_store_plugin_name, outputs=plugin_store_output_information
                )
                plugin_store_restart_button.click(restart)
            plugins = load_plugins(Path("./plugins"))
            for plugin_name, plugin_module in plugins.items():
                if hasattr(plugin_module, "plugin"):
                    plugin_module.plugin()
            with gr.Tab("配置設定"):
                update_anr_button = gr.Button("更新 ANR")
                with gr.Row():
                    setting_modify_button = gr.Button("儲存")
                    setting_restart_button = gr.Button("重啟")
                    setting_restart_button.click(restart)
                setting_output_information = gr.Textbox(show_label=False, visible=False)
                token = gr.Textbox(
                    value=env.token,
                    label="Token",
                    lines=2,
                    visible=True if not env.share else False,
                )
                gr.Markdown(
                    "取得 Token 的方法: [**自述檔案**](https://github.com/zhulinyv/Semi-Auto-NovelAI-to-Pixiv#%EF%B8%8F-%E9%85%8D%E7%BD%AE)",
                    visible=True if not env.share else False,
                )
                format_input = gr.Checkbox(value=env.format_input, label="格式化輸入")
                gr.Markdown("啟用後，將對輸入的提示詞進行格式化（刪除多餘空格和逗號或新增缺少的空格和逗號）")
                proxy = gr.Textbox(value=env.proxy, label="代理位址")
                gr.Markdown("<p>本地代理格式應為: http://127.0.0.1:xxx (xxx 為代理軟體的連接埠號)</p>")
                custom_path = gr.Textbox(value=env.custom_path, label="自訂路徑")
                gr.Markdown(
                    "已支援的自動替換路徑: <類型>, <日期>, <種子>, <隨機字元>, <編號>, 推薦: `<類型>/<日期>/<種子>_<編號>`"
                )
                cool_time = gr.Slider(1, 600, env.cool_time, label="冷卻時間")
                gr.Markdown("會上下浮動 1 秒")
                port = gr.Textbox(value=env.port, label="ANR 的連接埠號")
                gr.Markdown("理論範圍：1 - 65535")
                share = gr.Checkbox(value=env.share, label="共享 Gradio 連結")
                gr.Markdown("產生一個有效期一週的可分享連結，可以在任意裝置上存取")
                with gr.Row():
                    start_sound = gr.Checkbox(value=env.start_sound, label="啟動提示音")
                    finish_sound = gr.Checkbox(value=env.finish_sound, label="完成提示音")
                check_update = gr.Checkbox(value=env.check_update, label="啟動時檢查更新")
                theme = gr.Dropdown(
                    value=env.theme,
                    choices=[
                        "無",
                        "gradio/base",
                        "gradio/glass",
                        "gradio/monochrome",
                        "gradio/seafoam",
                        "gradio/soft",
                        "gradio/dracula_test",
                        "abidlabs/dracula_test",
                        "abidlabs/Lime",
                        "abidlabs/pakistan",
                        "Ama434/neutral-barlow",
                        "dawood/microsoft_windows",
                        "finlaymacklon/smooth_slate",
                        "Franklisi/darkmode",
                        "freddyaboulton/dracula_revamped",
                        "freddyaboulton/test-blue",
                        "gstaff/xkcd",
                        "Insuz/Mocha",
                        "Insuz/SimpleIndigo",
                        "JohnSmith9982/small_and_pretty",
                        "nota-ai/theme",
                        "nuttea/Softblue",
                        "ParityError/Anime",
                        "reilnuud/polite",
                        "remilia/Ghostly",
                        "rottenlittlecreature/Moon_Goblin",
                        "step-3-profit/Midnight-Deep",
                        "Taithrah/Minimal",
                        "ysharma/huggingface",
                        "ysharma/steampunk",
                        "NoCrypt/miku",
                    ],
                    label="WebUI 主題",
                    allow_custom_value=True,
                )
                gr.Markdown(
                    f"[切換到淺色頁面](http://127.0.0.1:{env.port}/?__theme=light) [切換到深色頁面](http://127.0.0.1:{env.port}/?__theme=dark)"
                )
                setting_modify_button.click(
                    modify_env,
                    inputs=[
                        token,
                        proxy,
                        custom_path,
                        cool_time,
                        port,
                        share,
                        start_sound,
                        finish_sound,
                        check_update,
                        theme,
                        format_input,
                    ],
                    outputs=setting_output_information,
                )
                update_anr_button.click(
                    update_repo, inputs=gr.Textbox(BASE_PATH, visible=False), outputs=setting_output_information
                )

    model.change(
        update_components_for_models_change,
        inputs=model,
        outputs=[
            decrisp,
            sm,
            sm_dyn,
            legacy_uc,
            sampler,
            noise_schedule,
            undesired_contentc_preset,
            naiv4vibebundle_file,
            normalize_reference_strength_multiple,
            nai3vibe_column,
            character_reference_tab,
            naiv4vibebundle_file_instruction,
            furry_mode,
            character_position_tab,
        ],
    )
    sm.change(update_components_for_sm_change, inputs=sm, outputs=sm_dyn)
    sampler.change(update_components_for_sampler_change, inputs=sampler, outputs=noise_schedule)


anr.launch(
    inbrowser=True,
    share=env.share,
    server_port=env.port,
    favicon_path="./assets/logo.ico",
    allowed_paths=[f"{d}:" for d in string.ascii_uppercase if Path(f"{d}:").exists()],
)
