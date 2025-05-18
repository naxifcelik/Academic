import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
import pandas as pd
import csv
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from tqdm import tqdm
import argparse

"""
Hyper parameters
"""
TEXT_PROMPT = "kit_fox. English_setter. Siberian_husky. Australian_terrier. English_springer. grey_whale. lesser_panda. Egyptian_cat. ibex. Persian_cat. cougar. gazelle. porcupine. sea_lion. malamute. badger. Great_Dane. Walker_hound. Welsh_springer_spaniel. whippet. Scottish_deerhound. killer_whale. mink. African_elephant. Weimaraner. soft-coated_wheaten_terrier. Dandie_Dinmont. red_wolf. Old_English_sheepdog. jaguar. otterhound. bloodhound. Airedale. hyena. meerkat. giant_schnauzer. titi. three-toed_sloth. sorrel. black-footed_ferret. dalmatian. black-and-tan_coonhound. papillon. skunk. Staffordshire_bullterrier. Mexican_hairless. Bouvier_des_Flandres. weasel. miniature_poodle. Cardigan. malinois. bighorn. fox_squirrel. colobus. tiger_cat. Lhasa. impala. coyote. Yorkshire_terrier. Newfoundland. brown_bear. red_fox. Norwegian_elkhound. Rottweiler. hartebeest. Saluki. grey_fox. schipperke. Pekinese. Brabancon_griffon. West_Highland_white_terrier. Sealyham_terrier. guenon. mongoose. indri. tiger. Irish_wolfhound. wild_boar. EntleBucher. zebra. ram. French_bulldog. orangutan. basenji. leopard. Bernese_mountain_dog. Maltese_dog. Norfolk_terrier. toy_terrier. vizsla. cairn. squirrel_monkey. groenendael. clumber. Siamese_cat. chimpanzee. komondor. Afghan_hound. Japanese_spaniel. proboscis_monkey. guinea_pig. white_wolf. ice_bear. gorilla. borzoi. toy_poodle. Kerry_blue_terrier. ox. Scotch_terrier. Tibetan_mastiff. spider_monkey. Doberman. Boston_bull. Greater_Swiss_Mountain_dog. Appenzeller. Shih-Tzu. Irish_water_spaniel. Pomeranian. Bedlington_terrier. warthog. Arabian_camel. siamang. miniature_schnauzer. collie. golden_retriever. Irish_terrier. affenpinscher. Border_collie. hare. boxer. silky_terrier. beagle. Leonberg. German_short-haired_pointer. patas. dhole. baboon. macaque. Chesapeake_Bay_retriever. bull_mastiff. kuvasz. capuchin. pug. curly-coated_retriever. Norwich_terrier. flat-coated_retriever. hog. keeshond. Eskimo_dog. Brittany_spaniel. standard_poodle. Lakeland_terrier. snow_leopard. Gordon_setter. dingo. standard_schnauzer. hamster. Tibetan_terrier. Arctic_fox. wire-haired_fox_terrier. basset. water_buffalo. American_black_bear. Angora. bison. howler_monkey. hippopotamus. chow. giant_panda. American_Staffordshire_terrier. Shetland_sheepdog. Great_Pyrenees. Chihuahua. tabby.marmoset. Labrador_retriever. Saint_Bernard. armadillo. Samoyed. bluetick. redbone. polecat. marmot. kelpie. gibbon. llama. miniature_pinscher. wood_rabbit. Italian_greyhound. lion. cocker_spaniel. Irish_setter. dugong. Indian_elephant. beaver. Sussex_spaniel. Pembroke. Blenheim_spaniel. Madagascar_cat. Rhodesian_ridgeback. lynx. African_hunting_dog. langur. Ibizan_hound. timber_wolf. cheetah. English_foxhound. briard. sloth_bear. Border_terrier. German_shepherd. otter. koala. tusker. echidna. wallaby. platypus. wombat. accordion. ant. starfish. chambered_nautilus. grand_piano. laptop. airliner. airship. balloon. fireboat. gondola. lifeboat. canoe. catamaran. container_ship. liner. aircraft_carrier. half_track. missile. bobsled. dogsled. bicycle-built-for-two. mountain_bike. freight_car. barrow. motor_scooter. forklift. electric_locomotive. amphibian. ambulance. beach_wagon. cab. convertible. jeep. limousine. minivan. Model_T. go-kart. golfcart. moped. fire_engine. garbage_truck. moving_van. mobile_home. horse_cart. jinrikisha. oxcart. bassinet. cradle. crib. four-poster. bookcase. china_cabinet. medicine_chest. chiffonier. file. barber_chair. folding_chair. desk. dining_table. entertainment_center. organ. chime. drum. gong. maraca. marimba. banjo. cello. harp. acoustic_guitar. electric_guitar. cornet. French_horn. harmonica. ocarina. panpipe. bassoon. oboe. flute. hatchet. cleaver. letter_opener. lawn_mower. hammer. corkscrew. can_opener. chain_saw. cock. hen. ostrich. brambling. goldfinch. house_finch. junco. indigo_bunting. robin. bulbul. jay. magpie. chickadee. water_ouzel. kite. bald_eagle. vulture. great_grey_owl.black_grouse. ptarmigan. ruffed_grouse. prairie_chicken. peacock. quail. partridge. African_grey. macaw. sulphur-crested_cockatoo. lorikeet. coucal. bee_eater. hornbill. hummingbird. jacamar. toucan. drake. red-breasted_merganser. goose. black_swan. white_stork. black_stork. spoonbill. flamingo. American_egret. little_blue_heron. bittern. crane. limpkin. American_coot. bustard. ruddy_turnstone. red-backed_sandpiper. redshank. dowitcher. oystercatcher. European_gallinule. pelican. king_penguin. albatross. great_white_shark. tiger_shark. hammerhead. electric_ray. stingray. barracouta. coho. tench. goldfish. eel. rock_beauty. anemone_fish. lionfish. puffer. sturgeon. gar. loggerhead. leatherback_turtle. mud_turtle. terrapin. box_turtle. banded_gecko. common_iguana. American_chameleon. whiptail. agama. frilled_lizard. alligator_lizard. Gila_monster. green_lizard. African_chameleon. Komodo_dragon. triceratops. African_crocodile. American_alligator. thunder_snake. ringneck_snake. hognose_snake. green_snake. king_snake. garter_snake. water_snake. vine_snake. night_snake. boa_constrictor. rock_python. Indian_cobra. green_mamba. sea_snake. horned_viper. diamondback. sidewinder. European_fire_salamander. common_newt. eft. spotted_salamander. axolotl. bullfrog. tree_frog. tailed_frog. paintbrush. hand_blower. oxygen_mask. loudspeaker. microphone. mouse. electric_fan. oil_filter. guillotine. barometer. odometer. analog_clock. digital_clock. hourglass. digital_watch. magnetic_compass. binoculars. loupe. bow. cannon. assault_rifle. computer_keyboard. crane. lighter. abacus. cash_machine. desktop_computer. hand-held_computer. notebook. harvester. joystick. hook. car_wheel. paddlewheel. gas_pump. carousel. hard_disc. car_mirror. disk_brake. buckle. hair_slide. knot. combination_lock. padlock. nail. muzzle. candle. jack-o'-lantern. neck_brace. maypole. mousetrap. trilobite. harvestman. scorpion. black_and_gold_garden_spider. barn_spider. garden_spider. black_widow. tarantula. wolf_spider. tick. centipede. isopod. Dungeness_crab. rock_crab. fiddler_crab. king_crab. American_lobster. spiny_lobster. crayfish. hermit_crab. tiger_beetle. ladybug. ground_beetle. long-horned_beetle. leaf_beetle. dung_beetle. rhinoceros_beetle. weevil. fly. bee. grasshopper. cricket. walking_stick. cockroach. mantis. cicada. leafhopper. lacewing. dragonfly. damselfly. admiral. ringlet. monarch. cabbage_butterfly. sulphur_butterfly. lycaenid. jellyfish. sea_anemone. brain_coral. flatworm. nematode. conch. snail. slug. sea_slug. chiton. sea_urchin. sea_cucumber. iron. espresso_maker. microwave. Dutch_oven. dishwasher. Crock_Pot. frying_pan. caldron. coffeepot. altar. barn. greenhouse. palace. monastery. library. apiary. boathouse. church. mosque. cinema. home_theater. lumbermill. coil. obelisk. castle. grocery_store. bakery. barbershop. bookshop. butcher_shop. confectionery. fountain. cliff_dwelling. dock. brass. megalith. bannister. breakwater. dam. chainlink_fence. grille. mountain_tent. honeycomb. beacon. jean. carton. handkerchief. ashcan. necklace. croquet_ball. fur_coat. pajama. cocktail_shaker. chest. manhole_cover. modem. balance_beam. kimono. knee_pad. beer_bottle. crash_helmet. bottlecap. mask. maillot. football_helmet. bathing_cap. holster. golf_ball. feather_boa. cloak. drumstick. Christmas_stocking. hoopskirt. bonnet. baseball. face_powder. beer_glass. lampshade. bow_tie. mailbag. bucket. dishrag. mortar. paddle. chain. mixing_bowl. bulletproof_vest. drilling_platform. binder. cardigan. birdhouse. hamper. apron. backpack. bearskin. broom. mosquito_net. abaya. mortarboard. crutch. cuirass. military_uniform. lipstick. monitor. oscilloscope.mitten. brassiere. milk_can. envelope. miniskirt. cowboy_hat. bathtub. bullet_train. cassette. carpenter's_kit. ladle. lotion. hair_spray. academic_gown. dome. crate. chain_mail. barrel. ballpoint. basketball. bath_towel. cowboy_boot. gown. cellular_telephone. nipple. barbell. mailbox. lab_coat. fire_screen. minibus. packet. maze. horizontal_bar. cassette_player. bell_cote. fountain_pen. overskirt. bolo_tie. bib. measuring_cup. breastplate. goblet. dial_telephone. jersey. jigsaw_puzzle. diaper. Band_Aid. gasmask. doormat. Loafer. maillot. clog. iPod. matchstick. bikini. CD_player. lens_cap. beaker. flagpole. coffee_mug. dumbbell."
  # Modified to be more generic for various animal classes
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
OUTPUT_DIR = Path("outputs")
DUMP_JSON_RESULTS = True
DEFAULT_VALUE = 0
DATA_DIR = Path("/home/nax/Masaüstü/datasubset")
PREDICTIONS_CSV = "predictions.csv"
FINAL_CSV = "final_results.csv"

# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def initialize_model(device):
    # build SAM2 image predictor
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino model
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=device
    )

    return sam2_predictor, grounding_model

def process_image(img_path, sam2_predictor, grounding_model, device):
    base_name = Path(img_path).stem
    output_mask_file = OUTPUT_DIR / f"{base_name}_annotated_with_mask.jpg"
    output_json_file = OUTPUT_DIR / f"{base_name}_results.json"
    single_mask_file = OUTPUT_DIR / f"{base_name}_mask.png"

    # Continue processing as before
    image_source, image = load_image(img_path)
    sam2_predictor.set_image(image_source)
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device= "cpu"
    )

    # If no objects detected, return None
    if len(boxes) == 0:
        return None

    # Process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes.to(device) * torch.tensor([w, h, w, h], device=device)
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    confidences = confidences.cpu().numpy().tolist()
    class_names = labels

    # Return the class with highest confidence
    if len(class_names) > 0:
        best_idx = np.argmax(confidences)
        return class_names[best_idx]
    return None

def create_predictions_csv(sam2_predictor, grounding_model, device):
    # Create CSV file for predictions
    with open(PREDICTIONS_CSV, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Filename', 'PredictedClass'])
        
        # Process all images in the datasubset directory
        for img_file in tqdm(sorted(DATA_DIR.glob('*.JPEG'))):
            encrypted_class = img_file.stem  # Get filename without extension
            predicted_class = process_image(str(img_file), sam2_predictor, grounding_model, device)
            
            if predicted_class is None:
                predicted_class = "unknown"
                
            csvwriter.writerow([encrypted_class, predicted_class])
    
    print(f"Predictions saved to {PREDICTIONS_CSV}")

def create_final_csv(ground_truth_csv):
    # Load predictions
    predictions_df = pd.read_csv(PREDICTIONS_CSV)
    
    # Load ground truth
    ground_truth_df = pd.read_csv(ground_truth_csv)
    
    # Merge dataframes
    final_df = pd.merge(
        predictions_df, 
        ground_truth_df, 
        left_on='Filename', 
        right_on='EncryptedClass', 
        how='left'
    )
    
    # Rename columns for the final output
    final_df = final_df[['Filename', 'ActualClass', 'PredictedClass']]
    final_df.columns = ['EncryptedClass', 'ActualClass', 'PredictedClass']
    
    # Save to CSV
    final_df.to_csv(FINAL_CSV, index=False)
    print(f"Final results saved to {FINAL_CSV}")
    
    # Calculate accuracy
    correct = (final_df['ActualClass'] == final_df['PredictedClass']).sum()
    total = len(final_df)
    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")

def main():
    parser = argparse.ArgumentParser(description="Grounded SAM2 Image Processing")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU ID to use, -1 for CPU")
    parser.add_argument("--ground_truth", type=str, help="Path to ground truth CSV")
    args = parser.parse_args()

    if args.gpu >= 0:
        device = f"cuda:{args.gpu}"
    else:
        device = "cpu"

    sam2_predictor, grounding_model = initialize_model(device)

    # Process all images and create predictions CSV
    create_predictions_csv(sam2_predictor, grounding_model, device)
    
    # Create final CSV with ground truth if provided
    if args.ground_truth:
        create_final_csv(args.ground_truth)

if __name__ == "__main__":
    main()
