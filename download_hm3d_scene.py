#!/usr/bin/env python3
"""
Stream-download a single HM3D scene without pulling the full archive.

Works by piping curl → tarfile in streaming mode and stopping as soon
as the target scene's files have been extracted.

For scene 00033 (33rd of 800 train scenes) this downloads ~600 MB
instead of ~15 GB.

Usage:
    conda activate habitat
    python3 ~/download_hm3d_scene.py
"""
import getpass, glob, os, signal, subprocess, sys, tarfile

SCENE_ID = '00033-oPj9qMxrDEa'
OUT_DIR  = os.path.expanduser('~/habitat_data/scene_datasets/hm3d/train')

# Files to fetch.  GLB  = mesh/navmesh,  configs = scene instance JSON.
TARGETS = [
    ('GLB    ', 'https://api.matterport.com/resources/habitat/hm3d-train-glb-v0.2.tar'),
    # configs has NO version suffix — special-cased in habitat-sim downloader
    ('Configs', 'https://api.matterport.com/resources/habitat/hm3d-train-configs.tar'),
]


def check_auth(url: str, user: str, pw: str) -> int:
    """Return HTTP status code for a small range request."""
    result = subprocess.run(
        ['curl', '--location', '--user', f'{user}:{pw}',
         '--range', '0-511', '--silent',
         '--write-out', '%{http_code}',
         '--output', '/dev/null',
         '--max-time', '15', url],
        capture_output=True, text=True
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def stream_extract(label: str, url: str, user: str, pw: str, out_dir: str, scene_id: str):
    os.makedirs(out_dir, exist_ok=True)

    cmd  = ['curl', '--location', '--user', f'{user}:{pw}', '--silent', url]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    extracted = []
    bytes_read = 0
    tar_error  = None
    try:
        with tarfile.open(fileobj=proc.stdout, mode='r|') as tar:
            for member in tar:
                bytes_read += member.size

                if scene_id not in member.name:
                    continue

                name_short = member.name.split('/')[-1]
                print(f'  [{label}] {name_short}  ({member.size / 1e6:.1f} MB)', flush=True)
                tar.extract(member, path=out_dir)
                extracted.append(member.name)

                if member.name.endswith('.glb'):
                    break

    except (BrokenPipeError, tarfile.ReadError) as e:
        if not extracted:
            tar_error = e
    finally:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        stderr_out = proc.stderr.read().decode(errors='replace').strip()
        proc.wait()

    if tar_error and not extracted:
        if stderr_out:
            print(f'  [{label}] curl stderr: {stderr_out}', flush=True)
        raise RuntimeError(
            f'tarfile parse failed ({tar_error}). '
            'The server likely returned an error page instead of the archive.'
        )

    downloaded_mb = bytes_read / 1e6
    print(f'  [{label}] stream closed after {downloaded_mb:.0f} MB read', flush=True)
    return extracted


def main():
    print(f'Target scene : {SCENE_ID}')
    print(f'Output dir   : {OUT_DIR}\n')

    print('Credentials: Matterport API token (NOT email+password)')
    print('Get yours at: https://my.matterport.com/settings/account/devtools\n')
    user = input('API Token ID     : ').strip()
    pw   = getpass.getpass('API Token Secret : ')
    print()

    # Verify credentials before streaming
    print('Checking credentials...', flush=True)
    status = check_auth(TARGETS[0][1], user, pw)
    if status == 401:
        print(
            'ERROR: 401 Unauthorized.\n'
            'Make sure you are using:\n'
            '  username = API Token ID   (NOT your email)\n'
            '  password = API Token Secret\n\n'
            'Generate a token at: https://my.matterport.com/settings/account/devtools\n'
            'Note: the secret is only shown once at creation time.'
        )
        sys.exit(1)
    elif status not in (200, 206):
        print(f'ERROR: unexpected HTTP {status} from Matterport API.')
        sys.exit(1)
    print(f'Credentials OK (HTTP {status})\n')

    all_extracted = []
    for label, url in TARGETS:
        print(f'[{label}] Streaming {url.split("/")[-1]} ...')
        try:
            files = stream_extract(label, url, user, pw, OUT_DIR, SCENE_ID)
            if not files:
                print(f'  [{label}] WARNING: no files matching {SCENE_ID!r} found')
            for f in files:
                print(f'  [{label}] ✓  {f}')
            all_extracted.extend(files)
        except RuntimeError as e:
            print(f'  [{label}] SKIPPED ({e})')
            print(f'  [{label}] (configs are optional — GLB alone is enough for rendering)')
        print()

    scene_dir = os.path.join(OUT_DIR, SCENE_ID)
    glbs = glob.glob(os.path.join(scene_dir, '*.glb'))
    if glbs:
        print(f'Done!  Scene ready at: {scene_dir}')
        for f in sorted(os.listdir(scene_dir)):
            path = os.path.join(scene_dir, f)
            print(f'  {f}  ({os.path.getsize(path) / 1e6:.1f} MB)')
        print(f'\nRun: python3 ~/habitat_bridge_pub.py')
    else:
        print('ERROR: GLB not found — check your credentials or URL.')
        sys.exit(1)


if __name__ == '__main__':
    main()
