# BALL GUIDE

## 1. Deskripsi Game

Ball Guide adalah game interaktif berbasis computer vision yang menggunakan kamera untuk mendeteksi gerakan objek berwarna kuning, seperti sarung tangan. Objek tersebut berfungsi sebagai paddle untuk memantulkan bola menuju target.

Game ini dibuat menggunakan Python, OpenCV, dan NumPy. Pemain mengontrol permainan melalui gerakan tangan di depan kamera tanpa perlu menggunakan perangkat kontrol fisik untuk mekanisme utama permainan.

## 2. Sistem Game

### 2.1. Mekanisme Permainan

Game memiliki mekanisme sebagai berikut:

1. Pemain memulai permainan melalui tombol START atau keyboard.
2. Kamera mendeteksi objek berwarna kuning yang berfungsi sebagai paddle.
3. Bola bergerak mengikuti simulasi gravitasi dan kecepatan.
4. Bola memantul ketika mengenai dinding, obstacle, bumper, atau paddle.
5. Pemain mengarahkan bola menuju target berwarna hijau.
6. Setiap target yang terkena menambah skor dan menghasilkan target baru.
7. Pemain memiliki tiga nyawa. Nyawa berkurang ketika bola jatuh keluar dari area permainan.
8. Permainan berakhir ketika seluruh nyawa habis.

### 2.2. Simulasi Fisika Bola

Program menggunakan parameter gravitasi, koefisien pantulan, dan batas kecepatan untuk mengatur gerakan bola.

```python
W, H = 640, 480
R = 12
GRAVITY, BOUNCE, MAXV = 700, 0.65, 650
```

Keterangan:

- `W` dan `H` menentukan resolusi area permainan.
- `R` menentukan jari-jari bola.
- `GRAVITY` mengatur percepatan gravitasi.
- `BOUNCE` menentukan koefisien pantulan pada permukaan biasa.
- `MAXV` membatasi kecepatan maksimum bola.

Gravitasi dan posisi bola diperbarui berdasarkan selang waktu setiap frame.

```python
for _ in range(4):
    h = dt / 4
    v[1] += GRAVITY * h

    sp = np.hypot(*v)
    if sp > MAXV:
        v *= MAXV / sp

    p += v * h
```

Program membagi pembaruan fisika menjadi empat sub-langkah untuk membantu menangani tumbukan secara lebih konsisten.

### 2.3. Sistem Tumbukan

Fungsi `collide()` mendeteksi apakah bola bersentuhan dengan permukaan tertentu. Fungsi ini juga mengubah kecepatan bola ketika terjadi tumbukan.

```python
def collide(p, v, solid, vs, e=BOUNCE):
    x, y = int(p[0]), int(p[1])

    if x < R or y < R or x >= W - R or y >= H - R:
        return

    hit = solid[y - R:y + R + 1, x - R:x + R + 1] & DISK

    if hit.sum() < 4:
        return

    ys, xs = np.nonzero(hit)
    n = -np.array([xs.mean() - R, ys.mean() - R])
    d = np.hypot(*n)
    n = n / d if d > 1e-3 else np.array([0.0, -1.0])

    rel = v - vs
    vn = rel @ n

    if vn < 0:
        v[:] = vs + rel - (1 + e) * vn * n

    p += n * 2
```

Parameter `solid` merupakan mask permukaan yang diperiksa, sedangkan `vs` merupakan kecepatan permukaan tersebut. Parameter `e` menentukan koefisien pantulan.

Pada implementasi game, tumbukan diterapkan pada obstacle, bumper, dan paddle.

```python
collide(p, v, bars_m, np.zeros(2))
collide(p, v, bump_m, np.zeros(2), 1.25)
collide(p, v, solid_y, vp)
```

Bumper menggunakan koefisien pantulan `1.25` untuk menghasilkan pantulan yang lebih kuat. Paddle menggunakan `vp`, yaitu estimasi kecepatan objek kuning yang terdeteksi kamera.

---

## 3. Motion Tracking

Motion tracking merupakan proses melacak posisi dan pergerakan objek dari frame kamera secara berurutan. Pada Bounce On It, sistem melacak objek kuning yang telah dideteksi melalui proses pengolahan citra.

Hasil tracking digunakan untuk menentukan posisi paddle dan memperkirakan kecepatannya agar gerakan tangan dapat memengaruhi arah pantulan bola.

### 3.1. Pengambilan Gambar dari Kamera

Program menggunakan OpenCV untuk mengambil frame dari kamera.

```python
cap = cv2.VideoCapture(0)
```

Setiap frame kemudian dibalik secara horizontal dan diubah ukurannya menjadi 640 × 480 piksel.

```python
ok, frame = cap.read()

if not ok:
    print("Camera not found")
    break

frame = cv2.resize(
    cv2.flip(frame, 1),
    (W, H)
)
```

Pembalikan horizontal membuat tampilan kamera menyerupai gerakan pada cermin. Ukuran frame yang konsisten juga mempermudah perhitungan koordinat objek dan interaksi dengan elemen permainan.

### 3.2. Menentukan Posisi Objek

Setelah objek kuning berhasil dideteksi, program menghitung momen citra menggunakan `cv2.moments()` untuk memperoleh titik pusat atau centroid.

```python
solid_y = yellow > 0

if solid_y.any():
    m = cv2.moments(yellow, True)

    c = np.array([
        m["m10"] / m["m00"],
        m["m01"] / m["m00"]
    ])
```

Centroid dihitung menggunakan persamaan:

\[
x_c = \frac{M_{10}}{M_{00}}
\]

\[
y_c = \frac{M_{01}}{M_{00}}
\]

Keterangan:

- \(x_c\) adalah koordinat horizontal pusat objek.
- \(y_c\) adalah koordinat vertikal pusat objek.
- \(M_{10}\) dan \(M_{01}\) merupakan momen spasial citra.
- \(M_{00}\) merupakan momen orde nol yang menunjukkan luas area mask.

Koordinat centroid menjadi representasi posisi objek yang digunakan oleh sistem tracking.

### 3.3. Menghitung Kecepatan Objek

Program memperkirakan kecepatan paddle dengan membandingkan posisi centroid pada frame saat ini dan frame sebelumnya.

```python
vp = np.clip(
    0.6 * vp + 0.4 * (c - pc) / dt,
    -900,
    900
) if pc is not None else np.zeros(2)

pc = c
```

Kecepatan dasar dihitung menggunakan persamaan:

\[
v_x = \frac{x_t-x_{t-1}}{\Delta t}
\]

\[
v_y = \frac{y_t-y_{t-1}}{\Delta t}
\]

Program menggabungkan 60% estimasi kecepatan sebelumnya dan 40% hasil pengukuran terbaru. Metode ini membantu mengurangi perubahan kecepatan yang terlalu mendadak.

Fungsi `np.clip()` membatasi setiap komponen kecepatan pada rentang -900 hingga 900 piksel per detik.

### 3.4. Penanganan Objek yang Tidak Terdeteksi

Ketika mask tidak berisi objek yang terdeteksi, program menghapus posisi tracking sebelumnya dan mengatur kecepatan paddle menjadi nol.

```python
else:
    pc, vp = None, np.zeros(2)
```

Langkah ini mencegah penggunaan posisi lama untuk memperkirakan gerakan objek yang sudah tidak terlihat oleh kamera.

### 3.5. Penggunaan Motion Tracking pada Game

Kecepatan paddle hasil tracking diteruskan ke fungsi tumbukan bola.

```python
collide(p, v, solid_y, vp)
```

Dengan demikian, bola tidak hanya memantul berdasarkan permukaan paddle. Gerakan paddle juga memengaruhi kecepatan relatif saat terjadi tumbukan.

---

## 4. Pendeteksi Warna

Pendeteksi warna digunakan untuk mengenali objek kuning dari gambar kamera. Sistem tidak melakukan pengenalan objek berdasarkan bentuk atau identitas, tetapi memilih piksel yang berada dalam rentang warna tertentu.

Program menggunakan ruang warna HSV dan fungsi `cv2.inRange()` untuk melakukan deteksi.

### 4.1. Konfigurasi Rentang HSV

Rentang warna awal ditentukan melalui variabel berikut.

```python
HSV = {
    "H min": 10,
    "H max": 35,
    "S min": 60,
    "S max": 255,
    "V min": 80,
    "V max": 255
}
```

HSV memiliki tiga komponen:

- **Hue (H):** Menentukan corak warna.
- **Saturation (S):** Menentukan tingkat kejenuhan warna.
- **Value (V):** Menentukan tingkat kecerahan warna.

Rentang awal tersebut digunakan untuk mendeteksi warna kuning. Hasil deteksi bergantung pada kondisi pencahayaan, kamera, dan karakteristik warna objek.

### 4.2. Konversi Gambar ke HSV

Sebelum melakukan penyaringan warna, program menghaluskan gambar menggunakan Gaussian Blur dan mengubah format warnanya dari BGR menjadi HSV.

```python
hsv = cv2.cvtColor(
    cv2.GaussianBlur(frame, (7, 7), 0),
    cv2.COLOR_BGR2HSV
)
```

Gaussian Blur dengan kernel 7 × 7 membantu mengurangi variasi kecil dan gangguan visual sebelum proses deteksi warna dilakukan.

HSV dipilih karena memisahkan corak warna dari komponen kejenuhan dan kecerahan, sehingga rentang warna lebih mudah disesuaikan.

### 4.3. Proses Penyaringan Warna

Program mengambil nilai minimum dan maksimum HSV dari konfigurasi, lalu menggunakannya untuk menghasilkan mask awal.

```python
def hsv_range():
    return (
        np.array([
            HSV["H min"],
            HSV["S min"],
            HSV["V min"]
        ], np.uint8),

        np.array([
            HSV["H max"],
            HSV["S max"],
            HSV["V max"]
        ], np.uint8)
    )
```

Penyaringan warna dilakukan melalui fungsi berikut.

```python
lo, hi = hsv_range()
raw = cv2.inRange(hsv, lo, hi)
```

`cv2.inRange()` memeriksa setiap piksel gambar HSV. Piksel yang memenuhi seluruh batas H, S, dan V menjadi putih, sedangkan piksel yang berada di luar rentang menjadi hitam.

Hasilnya adalah raw mask yang menunjukkan lokasi piksel dengan warna yang sesuai.

### 4.4. Pengaturan Warna Secara Real-Time

Program menyediakan enam trackbar untuk menyesuaikan batas minimum dan maksimum HSV.

```python
HSV_KEYS = [
    ("H min", 179),
    ("H max", 179),
    ("S min", 255),
    ("S max", 255),
    ("V min", 255),
    ("V max", 255)
]
```

Trackbar dibuat menggunakan OpenCV.

```python
for k, mx in HSV_KEYS:
    cv2.createTrackbar(
        k,
        CTRL_WIN,
        HSV[k],
        mx,
        lambda x: None
    )
```

Nilai pengaturan dibaca kembali setiap kali sistem memproses frame saat jendela Controls aktif.

```python
def read_settings():
    for k, _ in HSV_KEYS:
        HSV[k] = cv2.getTrackbarPos(k, CTRL_WIN)
```

Fitur ini membantu menyesuaikan deteksi warna ketika kondisi pencahayaan berubah atau objek kuning sulit dipisahkan dari latar belakang.

### 4.5. Penyimpanan Konfigurasi HSV

Program menyimpan konfigurasi HSV ke file `hsv_config.json` sehingga pengaturan yang sudah disesuaikan dapat digunakan kembali.

```python
def load_hsv():
    try:
        with open(HSV_FILE) as f:
            HSV.update({
                k: int(v)
                for k, v in json.load(f).items()
                if k in HSV
            })
    except Exception:
        pass
```

Fungsi penyimpanan:

```python
def save_hsv():
    try:
        with open(HSV_FILE, "w") as f:
            json.dump(HSV, f)
    except Exception:
        pass
```

Program memuat konfigurasi ketika permainan dimulai dan menyimpannya ketika pengaturan ditutup atau permainan berakhir.

---

## 5. Masking

Masking merupakan proses memisahkan area gambar berdasarkan kriteria tertentu. Pada game ini, masking digunakan untuk mengisolasi piksel objek kuning sehingga posisi objek dapat dihitung dan digunakan sebagai paddle.

Proses masking dilakukan dalam fungsi `find_yellow()`.

### 5.1. Implementasi Fungsi Masking

Berikut kode utama pengolahan citra pada program.

```python
def find_yellow(frame):
    """Camera frame -> (raw yellow mask, cleaned yellow mask)."""

    lo, hi = hsv_range()

    hsv = cv2.cvtColor(
        cv2.GaussianBlur(frame, (7, 7), 0),
        cv2.COLOR_BGR2HSV
    )

    raw = cv2.inRange(hsv, lo, hi)

    clean = cv2.morphologyEx(
        raw,
        cv2.MORPH_OPEN,
        K5,
        iterations=2
    )

    clean = cv2.morphologyEx(
        clean,
        cv2.MORPH_CLOSE,
        K5,
        iterations=2
    )

    return raw, (
        clean if clean.sum() > 255 * 800
        else np.zeros_like(clean)
    )
```

Fungsi tersebut menghasilkan dua keluaran:

- `raw`: Mask awal hasil penyaringan warna.
- `clean`: Mask yang sudah dibersihkan dan lolos pemeriksaan luas minimum.

### 5.2. Raw Mask

Raw mask merupakan hasil langsung dari proses thresholding warna.

```python
raw = cv2.inRange(hsv, lo, hi)
```

Pada tahap ini, setiap piksel yang memenuhi rentang HSV ditandai sebagai putih. Piksel lain menjadi hitam.

Raw mask dapat mengandung noise, seperti titik putih kecil atau area lain yang memiliki warna serupa dengan objek utama. Oleh karena itu, mask perlu dibersihkan sebelum digunakan untuk tracking.

### 5.3. Morphological Opening

Program menerapkan operasi morphological opening menggunakan kernel 5 × 5.

```python
K5 = np.ones((5, 5), np.uint8)
```

```python
clean = cv2.morphologyEx(
    raw,
    cv2.MORPH_OPEN,
    K5,
    iterations=2
)
```

Opening melakukan erosi yang diikuti dilasi. Operasi ini membantu menghilangkan noise putih berukuran kecil yang terpisah dari area utama objek.

Parameter `iterations=2` berarti operasi dijalankan sebanyak dua iterasi.

### 5.4. Morphological Closing

Setelah opening, program menerapkan morphological closing.

```python
clean = cv2.morphologyEx(
    clean,
    cv2.MORPH_CLOSE,
    K5,
    iterations=2
)
```

Closing melakukan dilasi yang diikuti erosi. Operasi ini membantu menutup lubang kecil dan menyambungkan bagian objek yang terputus.

Kombinasi opening dan closing menghasilkan mask yang lebih bersih untuk digunakan dalam perhitungan centroid dan tumbukan.

### 5.5. Penyaringan Berdasarkan Luas Mask

Program memeriksa luas area putih sebelum menerima hasil mask sebagai objek yang valid.

```python
return raw, (
    clean if clean.sum() > 255 * 800
    else np.zeros_like(clean)
)
```

Setiap piksel putih memiliki nilai 255. Oleh karena itu, kondisi `clean.sum() > 255 * 800` berarti jumlah piksel putih harus lebih dari 800.

Jika luas mask tidak memenuhi batas tersebut, program menghasilkan mask kosong menggunakan `np.zeros_like(clean)`.

Pemeriksaan ini membantu mengurangi deteksi noise kecil sebagai paddle. Namun, filter luas saja tidak menjamin bahwa setiap area yang lolos merupakan sarung tangan, karena objek lain dengan warna serupa dan area cukup besar juga dapat terdeteksi.

### 5.6. Visualisasi Masking

Program menampilkan hasil masking melalui jendela khusus.

```python
raw, yellow = find_yellow(frame)

cv2.imshow(
    MASK_WIN,
    mask_view(frame, raw, yellow)
)
```

Fungsi `mask_view()` menampilkan mask dan gambar kamera yang telah diterapkan mask.

```python
def mask_view(frame, clean, final):
    area = int(clean.sum() // 255)

    m = cv2.cvtColor(
        cv2.resize(clean, (320, 240)),
        cv2.COLOR_GRAY2BGR
    )

    seen = cv2.resize(
        cv2.bitwise_and(frame, frame, mask=clean),
        (320, 240)
    )

    ok = final.any()

    cv2.putText(
        m,
        f"mask  area={area}px",
        (6, 20),
        FONT,
        0.5,
        (0, 255, 0) if ok else (0, 0, 255),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        m,
        "DETECTED" if ok else "TOO SMALL (<800px)",
        (6, 40),
        FONT,
        0.5,
        (0, 255, 0) if ok else (0, 0, 255),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        seen,
        "camera x mask",
        (6, 20),
        FONT,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return np.hstack([m, seen])
```

Keterangan:

- `area` menunjukkan jumlah piksel putih pada mask.
- `DETECTED` menunjukkan bahwa mask akhir memiliki area yang memenuhi ambang batas.
- `TOO SMALL` menunjukkan bahwa area mask belum memenuhi batas minimum.
- `cv2.bitwise_and()` mempertahankan bagian gambar yang sesuai dengan mask.
- `np.hstack()` menggabungkan tampilan mask dan hasil penerapannya secara horizontal.

Jendela ini membantu memeriksa apakah rentang HSV sudah sesuai dengan warna objek yang digunakan.

### 5.7. Alur Masking

```text
Frame Kamera
     |
     v
Gaussian Blur
     |
     v
Konversi BGR ke HSV
     |
     v
Thresholding dengan cv2.inRange()
     |
     v
Raw Mask
     |
     v
Morphological Opening
     |
     v
Morphological Closing
     |
     v
Pemeriksaan Luas Mask
     |
     v
Clean Mask
     |
     v
Centroid dan Motion Tracking
     |
     v
Paddle untuk Interaksi dengan Bola
```

---

## 6. Teknologi yang Digunakan

| Teknologi | Fungsi |
|---|---|
| Python | Bahasa pemrograman utama |
| OpenCV | Pengambilan frame kamera, pengolahan citra, deteksi warna, dan tampilan game |
| NumPy | Operasi array, mask, centroid, kecepatan, dan perhitungan fisika |
| Random | Menghasilkan posisi target, obstacle, dan bumper secara acak |
| Time | Menghitung selang waktu untuk simulasi dan interaksi |
| Deque | Menyimpan posisi bola untuk membuat efek jejak |
| JSON | Membaca dan menyimpan konfigurasi HSV |

## 7. Kesimpulan

Bounce On It menggabungkan simulasi fisika dan computer vision untuk menghasilkan permainan yang dikontrol melalui gerakan tangan. Sistem pendeteksi warna HSV memilih piksel kuning dari frame kamera, sedangkan masking membersihkan hasil deteksi melalui operasi morfologi dan penyaringan luas area. Selanjutnya, motion tracking menghitung centroid dan memperkirakan kecepatan objek untuk mengendalikan paddle. Ketiga proses ini bekerja bersama agar pergerakan tangan dapat memengaruhi tumbukan dan arah bola secara real-time.
