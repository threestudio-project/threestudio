import json

att3d_gallery_names = [
    "a_zoomed_out_DSLR_photo_of_a_baby_#_sitting_on_top_of_a_stack_of_pancakes",
]

subjects = [
    "pig",
    "bunny",
    "tiger",
    "goose",
    "mouse",
    "chimpanzee",
    "corgi",
    "humanoid",
    "squirrel"
]

def main():
    prompts = []
    for theme in att3d_gallery_names:
        for subject in subjects:
            prompts.append(theme.replace("#", subject).replace("_", " "))
    prompt_library = {
        "compositional_prompts": prompts
    }
    with open("load/composite_prompts_2.json", "w") as f:
        json.dump(prompt_library, f, indent=2)


if __name__ == "__main__":
    main()
