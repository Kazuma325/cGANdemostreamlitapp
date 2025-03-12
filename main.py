import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator の定義（保存時と同じ）
class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, img_size=28):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 128 * self.init_size ** 2)
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat((noise, label_input), -1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# モデルのロード（weights_only=False）
latent_dim = 10
n_classes = 10
generator = torch.load("GANgenerator.pth", map_location=device, weights_only=False)
generator.to(device)
generator.eval()

# Streamlit アプリ
st.title("GANによる手書き数字の生成")

# ユーザー入力（ラベル選択）
target_label = st.selectbox("生成する数字を選択してください:", list(range(n_classes)))

# 画像生成関数
def generate_image(generator, device, latent_dim, target_label):
    label_tensor = torch.tensor([target_label], dtype=torch.long, device=device)
    z = torch.randn(1, latent_dim, device=device)
    with torch.no_grad():
        gen_img = generator(z, label_tensor)
    gen_img = (gen_img + 1) / 2  # スケール調整
    return gen_img.cpu().squeeze().numpy()

# 画像生成ボタン
if st.button("画像を生成"):
    generated_image = generate_image(generator, device, latent_dim, target_label)
    
    # 画像表示
    fig, ax = plt.subplots()
    ax.imshow(generated_image, cmap='gray')
    ax.axis('off')
    st.pyplot(fig)
