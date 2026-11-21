"""Read the original GroupLens archive locally. Return movie-level stars, features and coverage."""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import re
import sys
import unicodedata
from zipfile import ZipFile

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from design import Data

SHA = "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"
URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def order_hash(keys, prefix):
    return sorted(keys, key=lambda k:sha256((prefix+str(k)).encode()).digest())


def title_key(title):
    return " ".join(re.findall(r"\w+",unicodedata.normalize("NFKC",title).casefold()))


def load_archive(path, panel):
    assert sha256(Path(path).read_bytes()).hexdigest() == SHA
    with ZipFile(path) as archive:
        def lines(name):
            return archive.read("ml-100k/"+name).decode("latin1").splitlines()
        genre = [line.split("|")[0] for line in lines("u.genre") if line]
        features = np.array(["release_year",*genre])
        id_to_key = {}
        movies = defaultdict(list)
        for line in lines("u.item"):
            fields = line.split("|")
            key = title_key(fields[1])
            id_to_key[int(fields[0])] = key
            movies[key].append((int(fields[2][-4:]) if fields[2] else None,
                                np.array(fields[5:],dtype=int)))
        keys = sorted(movies)
        held = set(order_hash(keys,"corrce23-movie|")[:400])
        assert len(keys) == 1664 and len(id_to_key) == 1682
        observed = {}
        voters = set()
        for row in lines("u.data"):
            user, catalog, rating, timestamp = map(int,row.split())
            key = id_to_key[catalog]
            voters.add(user)
            previous = observed.get((key,user))
            if previous is None or (timestamp,-catalog)>previous[:2]:
                observed[key,user] = timestamp,-catalog,rating
    assert len(voters) == 943 and len(genre) == 19
    test_voters = set(order_hash(voters,"corrce23-user|")[:472])
    hist = defaultdict(lambda: np.zeros(5,dtype=int))
    train_seen, test_seen = set(),set()
    for (key,user), (_,_,rating) in observed.items():
        assert 1 <= rating <= 5
        if panel == "disjoint_users" and ((key in held) != (user in test_voters)):
            continue
        hist[key][rating-1] += 1
        (test_seen if key in held else train_seen).add(user)
    keys = [k for k in keys if k in hist]
    counts = np.array([hist[k] for k in keys])
    X = []
    for key in keys:
        rows = movies[key]
        years = set(y for y,_ in rows if y is not None)
        assert len(years) <= 1
        X.append([next(iter(years),np.nan),*np.max([v for _,v in rows],axis=0)])
    X = np.array(X,dtype=float)
    train = np.array([i for i,k in enumerate(keys) if k not in held])
    test = np.array([i for i,k in enumerate(keys) if k in held])
    missing_year_count = int(np.isnan(X[:,0]).sum())
    X[np.isnan(X[:,0]),0] = np.nanmedian(X[train,0])
    assert not {keys[i] for i in train}.intersection(keys[i] for i in test)
    assert np.all(counts.sum(1)>0) and np.isfinite(X).all()
    overlap = len(train_seen & test_seen)
    if panel == "disjoint_users":
        assert overlap == 0
    else:
        assert len(test) == 400 and len(train) == 1264
    info = dict(panel=panel,original_movies=1682,grouped_movies=1664,original_votes=100000,
        deduplicated_votes=len(observed),deduplicated_user_movie_events_removed=100000-len(observed),
        train_movies=len(train),test_movies=len(test),assigned_test_movies=400,uncovered_test_movies=400-len(test),
        train_users=len(train_seen),test_users=len(test_seen),shared_users=overlap,
        train_votes=int(counts[train].sum()),test_votes=int(counts[test].sum()),missing_year_imputed=missing_year_count)
    for label,indices in (("train",train),("test",test)):
        n=counts[indices].sum(1)
        for metric,fn in (("min",np.min),("median",np.median),("max",np.max)):
            info[f"{label}_votes_per_movie_{metric}"]=float(fn(n))
    ids=np.array(keys)
    return Data("movielens_"+panel,X,counts,ids,ids,features,None,None,train,test),info
