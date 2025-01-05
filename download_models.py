from pyannote.audio import Pipeline

# Specify the model you want to download
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization',use_auth_token='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')

# Save the model locally
#pipeline.save_pretrained("./models/pyannote_speaker_diarization")

# Save the model locally (specific to components, not the full pipeline)
#pipeline.segmentation.model.save_pretrained("./models/segmentation")


print("Model downloaded and saved locally at './models/pyannote_speaker_diarization'")
