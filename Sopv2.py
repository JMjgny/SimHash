import hashlib

def md5_hash(image_path):
    with open(image_path, 'rb') as f:
        data = f.read()
    return hashlib.md5(data).hexdigest()

hash_original = md5_hash("DataSet/Authentic/beforephoto.jpg")
hash_modified = md5_hash("DataSet/Fraud/unblurphoto.jpg")
hash_heavy = md5_hash("DataSet/Fraud/remini.jpg")

print("Original:", hash_original)
print("Slightly Modified:", hash_modified)
print("Heavily Modified:", hash_heavy)