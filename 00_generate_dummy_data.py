import os
import numpy as np
import soundfile as sf

def generate_tone(freq, duration, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), False)
    tone = np.sin(freq * t * 2 * np.pi)
    return tone

def generate_noise(duration, sr=22050):
    return np.random.randn(int(sr * duration))

def generate_dummy_data(base_dir="dummy_audio", samples_per_class=20):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    classes = {
        "bird": lambda: generate_tone(np.random.uniform(3000, 5000), duration=2.0) + generate_noise(2.0)*0.1,
        "wind": lambda: generate_noise(duration=2.0) * 0.5,
        "machinery": lambda: generate_tone(np.random.uniform(50, 150), duration=2.0) + generate_tone(np.random.uniform(200, 300), duration=2.0)*0.5
    }

    print(f"Generating dummy data in {base_dir}...")
    for class_name, gen_func in classes.items():
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        for i in range(samples_per_class):
            audio = gen_func()
            # Normalize
            audio = audio / np.max(np.abs(audio))

            filepath = os.path.join(class_dir, f"{class_name}_{i:03d}.wav")
            sf.write(filepath, audio, 22050)
    print("Done generating dummy data.")

if __name__ == "__main__":
    generate_dummy_data()