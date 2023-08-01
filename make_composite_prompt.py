import json


activities = [
    "riding a bicycle",
    "sitting on a chair",
    "playing the guitar",
    "holding a shovel",
    "holding a blue balloon",
    "holding a book",
    "wielding a katana",
    "riding a bike"
]

themes = [
    "made out of gold",
    "carved out of wood",
    "wearing a leather jacket",
    "wearing a tophat",
    "wearing a party hat",
    "wearing a sombrero",
    "wearing medieval armor"
]

subjects = [
    "pig",
    # "bunny",
    # "tiger",
    # "goose",
    # "mouse",
    # "chimpanzee",
    # "corgi",
    # "humanoid",
    # "squirrel"
]


def main():
    prompts = []
    for theme in themes:
        for act in activities:
            for subject in subjects:
                prompts.append(f"a {subject} {act} {theme}")
    prompt_library = {
        "compositional_prompts": prompts
    }
    with open("load/composite_prompts_3.json", "w") as f:
        json.dump(prompt_library, f, indent=2)


if __name__ == "__main__":
    main()
