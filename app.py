from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
import pydicom
import numpy as np
from PIL import Image
from ecg import ECG
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt
from io import BytesIO


# Monkey patch for backward compatibility
pydicom.read_file = pydicom.dcmread

app = FastAPI(title="DICOM ECG Converter")
LISTEN_PORT = 4430


@app.post("/api/image")
async def convert_image_to_plot(file: UploadFile = File(...)):

    try:
        contents = await file.read()

        if file.filename.endswith(".hl7vector"):
            ecg_array = np.load(BytesIO(contents))
            png_bytes = plot_ecg_clinical(ecg_array)
            return Response(content=png_bytes, media_type="image/png")

        elif file.filename.endswith(".dcm"):
            file_obj = BytesIO(contents)
            ecg = ECG(file_obj)
            
            ecg.draw("6x2", 10, minor_axis=False, interpretation=False)
            png_bytes = ecg.save(outformat="png")

            return Response(content=png_bytes, media_type="image/png")

        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {e}")

def plot_ecg_clinical(ecg_array, sampling_rate=500):
    if ecg_array.shape[0] != 12:
        raise ValueError("Expected ECG data with 12 leads (shape [12, N])")

    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]
    num_leads = len(lead_names)
    duration_sec = ecg_array.shape[1] / sampling_rate

    fig, axs = plt.subplots(6, 2, figsize=(12, 15), constrained_layout=True)
    axs = axs.flatten()

    time_axis = np.linspace(0, duration_sec, ecg_array.shape[1])

    for i in range(num_leads):
        signal = ecg_array[i]
        ax = axs[i]
        ax.plot(time_axis, signal, color="black", linewidth=1)

        ax.set_xlim(0, duration_sec)
        ax.set_ylim(-2, 2)
        ax.set_ylabel(lead_names[i], rotation=0, labelpad=30, fontsize=12, weight='bold')

        ax.set_xticks(np.arange(0, duration_sec, 0.2), minor=False)
        ax.set_xticks(np.arange(0, duration_sec, 0.04), minor=True)
        ax.set_yticks(np.arange(-2, 2.5, 0.5), minor=False)
        ax.set_yticks(np.arange(-2, 2.1, 0.1), minor=True)
        ax.grid(which='major', color='red', linestyle='-', linewidth=0.5)
        ax.grid(which='minor', color='red', linestyle=':', linewidth=0.3)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    fig.suptitle("12-lead ECG", fontsize=16, weight='bold')

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    plt.close(fig)

    return buf.getvalue()




if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=LISTEN_PORT)
