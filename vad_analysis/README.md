# Processing Pipeline

1. Step 1: URL Building

```
pythonurls = build_urls(langs=["es", "ar", "fr"])
```

Scans `lid_index.pkl` for tar files containing the difference language you want to retrieve. **Note: Each tar contains mixed languages, not just the target ones**

2. Step 2: WebDataset Creation

```
pythonwds.WebDataset(urls, shardshuffle=1000)
    .to_tuple("mp3", "__key__", "__url__")
    .map(decode_and_normalize_with_vad)
```

Streams audio files from tar archives (no full download). `shardshuffle=1000` randomizes tar file order. Each file is decoded as`(mp3_bytes, filename, tar_url)`

3. Step 3: VAD-Based Processing `(decode_and_normalize_with_vad)`

**Language Filtering:**

```
pythonlookup_key = (tar_number, filename)
language = tar_to_lang.get(lookup_key, "unknown")
if desired_languages and language not in desired_languages:
    return None  # Skip non-target languages
```

**VAD Segment Loading:**

```
pythonvad_shard = load_vad_shard(tar_number)  # Loads cached pickle
vad_data = vad_shard.get(filename)
segments = vad_data.get('segments', [])  # [(start, end), ...]
Chunk Extraction:
python# Find all speech segments ≥10 seconds
candidate_windows = []
for seg_start, seg_end in segments:
    if seg_end - seg_start >= chunk_sec:
        candidate_windows.append((seg_start, seg_end - chunk_sec))
```

**Extract up to 16 random 10-second chunks**
```
for _ in range(num_chunks):
    seg_start, max_start = random.choice(candidate_windows)
    start_sec = random.uniform(seg_start, max_start)
    end_sec = start_sec + chunk_sec
    output_chunks.append(_stream_chunk(start_sec, end_sec))
```

**Skip Conditions:**

* No VAD data available
* No speech segments detected
* No segments ≥10 seconds long
* Language not in target list

4. Step 4: Batch Collation `(collate_fn)`

```
input_values = torch.cat([sample["input_values"] for sample in batch], dim=0)
```

**Track language for each chunk**
```
languages = []
for sample in batch:
    num_chunks = sample["input_values"].shape[0]
    languages.extend([sample["language"]] * num_chunks)
```

**The usage example is `main_example.py`**