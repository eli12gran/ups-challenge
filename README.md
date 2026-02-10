# VAD-Enhanced WebDataset Dataloader

**The `vad_analysis` directory must be added to the `/ups-challenge-baselines/ups_challenge` folder of the original dataloader** in order to work correctly

This dataloader builds upon the original UPS Challenge dataloader by integrating Voice Activity Detection (VAD) to ensure audio chunks contain actual speech. It filters audio by language and only extracts segments where speech is detected.

**Dependencies**
```
pip install webdataset torchaudio torchcodec
```

**Required Files**

1. `vad_results.jsonl` - Download and place in `/ups-submit/ups-challenge/ups-challenge-baselines/data`. Contains VAD timestamps for all audio files. Can be deleted after running `vad_lookup.py`

2. `lang_id_results.jsonl` - Automatically downloaded by the dataloader. Contains language identification results used to build `lid_index.pkl`

3. HuggingFace Token - Set as environment variable:

```
export HF_TOKEN=your_token_here
```
---

**Setup Instructions**
1. Build VAD Index (One-time setup). Run `vad_lookup.py` to process `vad_results.jsonl` into efficient sharded pickle files:

- Reads `vad_results.jsonl` (must be in `./data/`)
- Extracts speech segments with timestamps
- Creates sharded pickle files in `./data/vad_shards/` (one per tar file)
- Processes ~2.1M audio files

Output:

```
./data/vad_shards/
├── 000001.pkl
├── 000002.pkl
├── 000003.pkl
```

**After completion: You can safely delete `vad_results.jsonl` to save disk space.**

---

### **Architecture Overview**

1. build_urls() → Finds tar files containing target languages
                ↓
2. WebDataset   → Streams audio files from tar archives
                ↓
3. VAD Filter   → Checks if audio has speech segments ≥10s
                ↓
4. Language Filter → Keeps only desired languages
                ↓
5. Chunk Extraction → Extracts random 10s chunks from speech regions
                ↓
6. DataLoader   → Batches chunks for training