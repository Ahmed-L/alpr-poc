from ultralytics import YOLO

english_to_bangla = {
    "0": "০", "1": "১", "2": "২", "3": "৩", "4": "৪", "5": "৫", "6": "৬", "7": "৭", "8": "৮", "9": "৯",
    "A": "এ", "Ba": "বা", "Bagerhat": "বাগেরহাট", "Bagura": "বগুড়া", "Bandarban": "বান্দরবান",
    "Barguna": "বরগুনা", "Barisal": "বরিশাল", "Bha": "ভা", "Bhola": "ভোলা", "Brahmanbaria": "ব্রাহ্মণবাড়িয়া",
    "Cha": "চা", "Chandpur": "চাঁদপুর", "Chapainawabganj": "চাঁপাইনবাবগঞ্জ", "Chatto": "চট্ট",
    "Chattogram": "চট্টগ্রাম", "Chha": "ছা", "Chuadanga": "চুয়াডাঙ্গা", "Cox-s Bazar": "কক্সবাজার",
    "Cumilla": "কুমিল্লা", "DA": "ডা", "Da": "দা", "Dha": "ঢা", "Dhaka": "ঢাকা", "Dinajpur": "দিনাজপুর",
    "E": "ই", "Faridpur": "ফরিদপুর", "Feni": "ফেনী", "Ga": "গা", "Gaibandha": "গাইবান্ধা", "Gazipur": "গাজীপুর",
    "Gha": "ঘা", "Gopalganj": "গোপালগঞ্জ", "Ha": "হা", "Habiganj": "হবিগঞ্জ", "Ja": "জা", "Jamalpur": "জামালপুর",
    "Jessore": "যশোর", "Jha": "ঝা", "Jhalokati": "ঝালকাঠি", "Jhenaidah": "ঝিনাইদহ", "Joypurhat": "জয়পুরহাট",
    "Ka": "কা", "Kha": "খা", "Khagrachari": "খাগড়াছড়ি", "Khulna": "খুলনা", "Kishoreganj": "কিশোরগঞ্জ",
    "Kurigram": "কুড়িগ্রাম", "Kustia": "কুষ্টিয়া", "La": "লা", "Lakshmipur": "লক্ষ্মীপুর",
    "Lalmonirhat": "লালমনিরহাট", "Ma": "মা", "Madaripur": "মাদারীপুর", "Magura": "মাগুরা",
    "Manikganj": "মানিকগঞ্জ", "Meherpur": "মেহেরপুর", "Metro": "মেট্রো", "Moulvibazar": "মৌলভীবাজার",
    "Mymensingh": "ময়মনসিংহ", "Na": "না", "Naogaon": "নওগাঁ", "Narail": "নড়াইল", "Narayanganj": "নারায়ণগঞ্জ",
    "Narsingdi": "নরসিংদী", "Natore": "নাটোর", "Netrokona": "নেত্রকোনা", "Nilphamari": "নীলফামারী",
    "Noakhali": "নোয়াখালী", "Pa": "পা", "Pabna": "পাবনা", "Patuakhali": "পটুয়াখালী", "Pirojpur": "পিরোজপুর",
    "Raj": "রাজ", "Rajbari": "রাজবাড়ি", "Rajshahi": "রাজশাহী", "Rangamati": "রাঙামাটি",
    "Rangpur": "রংপুর", "Sa": "সা", "Satkhira": "সাতক্ষীরা", "Sha": "শা", "Shariatpur": "শরীয়তপুর",
    "Sherpur": "শেরপুর", "Sirajganj": "সিরাজগঞ্জ", "Sunamganj": "সুনামগঞ্জ", "Sylhet": "সিলেট",
    "THA": "ঠা", "Ta": "টা", "Tangail": "টাঙ্গাইল", "Tha": "থা", "Thakurgaon": "ঠাকুরগাঁও",
    "U": "উ", "panchagarh": "পঞ্চগড়"
}


model = YOLO("./config/model/ocr_yolov8s_best_v0001.pt")

results = model.predict("./saved_images/cropped_image_19.jpg", imgsz=(640, 480))



predictions = results[0]

# Extract bounding boxes, class IDs, and confidence scores
boxes = predictions.boxes.xyxy
class_ids = predictions.boxes.cls
confidences = predictions.boxes.conf

# Map class IDs to labels (license plate characters)
detected_chars = [
    {"char": model.names[int(cls_id)], "box": box.tolist()}
    for cls_id, box in zip(class_ids, boxes)
]

# Sort detections by x_min (leftmost first), and for ties, sort by y_min (topmost)
sorted_chars = sorted(detected_chars, key=lambda x: (x["box"][0], x["box"][1]))


letters = []
numbers = []

for char in sorted_chars:
    if char["char"].isdigit():
        numbers.append(english_to_bangla[char["char"]] + " ")
    else:
        letters.append(english_to_bangla[char["char"]] + " ")

# Join the letters and numbers
license_plate_text = "".join(letters)
numbers_text = "".join(numbers)

# Combine into final output
final_license_plate = license_plate_text
if numbers_text:  # Add numbers on a new line if they exist
    final_license_plate += f"\n{numbers_text}"

print("Detected License Plate:")
print(final_license_plate)