from main import VOLUME_CONTAINER

def first_fit_volume_decreasing(preprocessed):
    container = []
    volume_container = VOLUME_CONTAINER
    for idx, row in preprocessed.sort_values('volume', ascending=False).iterrows():
        volume = row['volume']
        quantidade = int(row['quantidade'])
        while quantidade > 0 and volume <= volume_container:
            quantidade -= 1
            volume_container -= volume
            container.append((idx, volume))
            print(f"Added box {idx} with volume {volume}, remaining volume in container: {volume_container}")
    return len(container), container

def first_fit_volume_crescente(preprocessed):
    container = []
    volume_container = VOLUME_CONTAINER
    for idx, row in preprocessed.sort_values('volume', ascending=True).iterrows():
        volume = row['volume']
        quantidade = int(row['quantidade'])
        while quantidade > 0 and volume <= volume_container:
            quantidade -= 1
            volume_container -= volume
            container.append((idx, volume))
            print(f"Added box {idx} with volume {volume}, remaining volume in container: {volume_container}")
    return len(container), container
