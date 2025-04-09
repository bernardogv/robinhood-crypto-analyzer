#!/usr/bin/env python3
"""
Key Generator for Robinhood Crypto API

This script generates an Ed25519 keypair and encodes it in base64 format
for use with the Robinhood Crypto API.
"""

import argparse
import base64
import nacl.signing
import os
import sys

def generate_keypair():
    """
    Generate an Ed25519 keypair for use with the Robinhood Crypto API.
    
    Returns:
        tuple: (private_key_base64, public_key_base64) - Base64-encoded keys
    """
    # Generate an Ed25519 keypair
    private_key = nacl.signing.SigningKey.generate()
    public_key = private_key.verify_key

    # Convert keys to base64 strings
    private_key_base64 = base64.b64encode(private_key.encode()).decode()
    public_key_base64 = base64.b64encode(public_key.encode()).decode()
    
    return private_key_base64, public_key_base64

def save_keys_to_file(private_key_base64, public_key_base64, output_dir):
    """
    Save the generated keys to files in the specified directory.
    
    Args:
        private_key_base64: Base64-encoded private key
        public_key_base64: Base64-encoded public key
        output_dir: Directory to save the key files
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the private key
    private_key_path = os.path.join(output_dir, "rh_private_key.b64")
    with open(private_key_path, "w") as f:
        f.write(private_key_base64)
    
    # Save the public key
    public_key_path = os.path.join(output_dir, "rh_public_key.b64")
    with open(public_key_path, "w") as f:
        f.write(public_key_base64)
    
    print(f"Private key saved to: {private_key_path}")
    print(f"Public key saved to: {public_key_path}")
    
    # Set appropriate permissions for the private key (readable only by the owner)
    try:
        os.chmod(private_key_path, 0o600)
        print(f"Set permissions on {private_key_path} to be readable only by the owner.")
    except Exception as e:
        print(f"Warning: Could not set permissions on {private_key_path}: {e}")
        print("Please ensure that your private key is stored securely.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate an Ed25519 keypair for the Robinhood Crypto API")
    parser.add_argument("--output-dir", "-o", default=".", 
                      help="Directory to save the key files (default: current directory)")
    parser.add_argument("--print", "-p", action="store_true",
                      help="Print the keys to stdout instead of saving to files")
    
    args = parser.parse_args()
    
    # Generate the keypair
    try:
        print("Generating Ed25519 keypair for Robinhood Crypto API...")
        private_key_base64, public_key_base64 = generate_keypair()
        
        if args.print:
            # Print the keys to stdout
            print("\nPrivate Key (Base64):")
            print(private_key_base64)
            
            print("\nPublic Key (Base64):")
            print(public_key_base64)
            
            print("\nIMPORTANT: Keep your private key secure and never share it!")
            print("You'll need to add the private key to your config.py file.")
            print("The public key is used when creating API credentials in the Robinhood API Credentials Portal.")
        else:
            # Save the keys to files
            save_keys_to_file(private_key_base64, public_key_base64, args.output_dir)
            
            print("\nIMPORTANT: Keep your private key secure and never share it!")
            print("You'll need to add the private key to your config.py file.")
            print("The public key is used when creating API credentials in the Robinhood API Credentials Portal.")
    
    except Exception as e:
        print(f"Error generating keypair: {e}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
