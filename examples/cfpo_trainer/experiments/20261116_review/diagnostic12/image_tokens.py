"""Unchanged image-token packing helper from successful fix2 (job224596)."""
def engine_prompt_ids(expanded_ids,image_id,vision_start_id,vision_end_id,image_counts):
    """HF expands image pads; vLLM token input needs one placeholder per image.

    Collapse only verified image runs, preserving every other token exactly.
    Counts must agree with the image grid, not merely with the number of images.
    """
    result=[];counts=[];ids=list(expanded_ids);i=0
    while i<len(ids):
        if ids[i]!=image_id:
            result.append(ids[i]);i+=1;continue
        start=i
        while i<len(ids) and ids[i]==image_id:i+=1
        if start==0 or ids[start-1]!=vision_start_id or i==len(ids) or ids[i]!=vision_end_id:
            raise ValueError('Image token outside a real vision block')
        counts.append(i-start);result.append(image_id)
    if counts!=list(image_counts):raise ValueError('Image pads do not match processed image grid: '+str(counts)+' vs '+str(image_counts))
    return result
