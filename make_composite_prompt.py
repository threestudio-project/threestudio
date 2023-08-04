import json

themes = [
    "blue",
    "red",
    "green",
    "purple"
]

subjects = [
    "a squirrel",
    "an airplane",
    "a hamburger",
    "a pineapple"
]


def main():
    prompts = []
    for i, theme in enumerate(themes):
        for j, subject in enumerate(subjects):
            if i != j:
                prompts.append(f"{subject}, {theme}")
    prompt_library = {
        "compositional_prompts": prompts
    }
    with open("load/composite_prompts_3.json", "w") as f:
        json.dump(prompt_library, f, indent=2)


prompts = [
    "lib:orangutan_wheel",
    "lib:raccoon_astronaut",
    "lib:macarons",
    "lib:photo_corgi_selfie",
    "lib:dim",
    "lib:lion_newspaper",
    "lib:Michelangelo_dog",
    "lib:tiger_doctor",
    "lib:steam_engine",
    "lib:frog_sweater",
    "lib:humanoid_cello",
    "lib:opera_house",
    "lib:vehicle",
    "lib:chimpanzee_England",
    "lib:bunny_pancakes",
    "lib:fresh_bread",
    "lib:bulldozer_snow",
    "lib:Packard_car",
    "lib:Tower_Bridge",
    "lib:robot_dinosaur_chess",
    "lib:squirrel_gesturing_charts",
    "lib:peacock",
    "lib:couple",
    "lib:pirate",
    "lib:delicious_hamburger",
    "lib:pink_flower",
    "lib:English_castle"
]


def main2():
    prompt_library = {
        "compositional_prompts": prompts
    }
    with open("load/DF27.json", "w") as f:
        json.dump(prompt_library, f, indent=2)


if __name__ == "__main__":
    main()
