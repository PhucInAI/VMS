# RSA helper class for pycrypto
# Copyright (c) Dennis Lee
# Date 21 Mar 2017

# Description:
# Python helper class to perform RSA encryption, decryption, 
# signing, verifying signatures & keys generation

# Dependencies Packages:
# pycrypto 

# Documentation:
# https://www.dlitz.net/software/pycrypto/api/2.6/

# Sample usage:
'''
import rsa
from base64 import b64encode, b64decode

msg1 = "Hello Tony, I am Jarvis!"
msg2 = "Hello Toni, I am Jarvis!"
keysize = 2048
(public, private) = rsa.newkeys(keysize)
encrypted = b64encode(rsa.encrypt(msg1, private))
decrypted = rsa.decrypt(b64decode(encrypted), private)
signature = b64encode(rsa.sign(msg1, private, "SHA-512"))
verify = rsa.verify(msg1, b64decode(signature), public)

print(private.exportKey('PEM'))
print(public.exportKey('PEM'))
print("Encrypted: " + encrypted)
print("Decrypted: '%s'" % decrypted)
print("Signature: " + signature)
print("Verify: %s" % verify)
rsa.verify(msg2, b64decode(signature), public)
'''
from Crypto import Random
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Hash import SHA512, SHA384, SHA256, SHA, MD5
from Crypto.PublicKey import RSA
from Crypto.Signature import PKCS1_v1_5

hash = "SHA-256"


def newkeys(keysize):
    Random.atfork()
    random_generator = Random.new().read
    key = RSA.generate(keysize, random_generator)
    private, public = key, key.publickey()
    print(private.exportKey('PEM'))
    print(public.exportKey('PEM'))
    return public, private


def importKey(externKey):
    Random.atfork()
    return RSA.importKey(externKey)


def getpublickey(priv_key):
    Random.atfork()
    return priv_key.publickey()


def encrypt(message, pub_key):
    # RSA encryption protocol according to PKCS#1 OAEP
    Random.atfork()
    cipher = PKCS1_OAEP.new(pub_key)
    return cipher.encrypt(message)


def decrypt(ciphertext, priv_key):
    # RSA encryption protocol according to PKCS#1 OAEP
    Random.atfork()
    cipher = PKCS1_OAEP.new(priv_key)
    return cipher.decrypt(ciphertext)


def sign(message, priv_key, hashAlg="SHA-256"):
    Random.atfork()
    global hash
    hash = hashAlg
    signer = PKCS1_v1_5.new(priv_key)
    if (hash == "SHA-512"):
        digest = SHA512.new()
    elif (hash == "SHA-384"):
        digest = SHA384.new()
    elif (hash == "SHA-256"):
        digest = SHA256.new()
    elif (hash == "SHA-1"):
        digest = SHA.new()
    else:
        digest = MD5.new()
    digest.update(message)
    return signer.sign(digest)


def verify(message, signature, pub_key):
    Random.atfork()
    signer = PKCS1_v1_5.new(pub_key)
    if (hash == "SHA-512"):
        digest = SHA512.new()
    elif (hash == "SHA-384"):
        digest = SHA384.new()
    elif (hash == "SHA-256"):
        digest = SHA256.new()
    elif (hash == "SHA-1"):
        digest = SHA.new()
    else:
        digest = MD5.new()
    digest.update(message)
    return signer.verify(digest, signature)
