import hashlib
import time
import datetime
import logging
from typing import List, Dict, Any

class Block:
    def __init__(self, index, previous_hash, transactions, nonce=0, timestamp=None):
        self.index = index
        self.previous_hash = previous_hash
        self.transactions = transactions
        self.timestamp = timestamp or time.time()
        self.nonce = nonce
        self.hash = self.calculate_hash()
        
    def calculate_hash(self):
        """Calculate the hash of the block using SHA-256."""
        block_string = f"{self.index}{self.previous_hash}{self.transactions}{self.timestamp}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine(self, difficulty):
        """Mine the block by finding a nonce that meets the difficulty."""
        while not self.hash.startswith("0" * difficulty):
            self.nonce += 1
            self.hash = self.calculate_hash()

class Blockchain:
    def __init__(self, db_params, difficulty=2):
        self.chain = [self.create_genesis_block()]
        self.difficulty = difficulty
        self.pending_transactions = []
        self.db_params = db_params
        
    def calculate_merkle_root(self, transactions):
        """Calculate the Merkle root of a list of transactions."""
        if not transactions:
            return None
        
        hashes = [hashlib.sha256(str(tx).encode()).hexdigest() for tx in transactions]
        while len(hashes) > 1:
            new_hashes = []
            for i in range(0, len(hashes), 2):
                if i + 1 < len(hashes):
                    combined_hash = hashlib.sha256(hashes[i] + hashes[i + 1]).hexdigest()
                    new_hashes.append(combined_hash)
                else:
                    new_hashes.append(hashes[i])
            hashes = new_hashes
        return hashes[0]

    def create_genesis_block(self):
        """Create the first block in the chain (genesis block)."""
        return Block(0, "0", "Genesis Block")

    def add_block(self, block):
        """Add a new block to the chain after mining."""
        if self.is_valid_block(block, self.chain[-1]):
            block.merkle_root = self.calculate_merkle_root(block.transactions)
            self.chain.append(block)
            self.pending_transactions = []
            return True
        return False

    def mine_block(self, miner_address):
        """Mine a new block with pending transactions."""
        previous_block = self.chain[-1]
        new_block = Block(
            index=len(self.chain),
            previous_hash=previous_block.hash,
            transactions=self.pending_transactions
        )
        new_block.mine(self.difficulty)
        
        if self.add_block(new_block):
            # Reward the miner
            self.add_transaction({"from": "network", "to": miner_address, "amount": 10})
            return new_block
        return None

    def add_transaction(self, transaction):
        """Add a new transaction to the list of pending transactions."""
        self.pending_transactions.append(transaction)

    def is_valid_block(self, block, previous_block):
        """Check if a block is valid."""
        if previous_block.index + 1 != block.index:
            return False
        elif previous_block.hash != block.previous_hash:
            return False
        elif not self.is_valid_proof(block, self.difficulty):
            return False
        return True

    def is_valid_proof(self, block, difficulty):
        """Check if a proof (hash) is valid based on the difficulty."""
        return block.hash.startswith("0" * difficulty)

    def is_chain_valid(self):
        """Check if the blockchain is valid."""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            if not self.is_valid_block(current_block, previous_block):
                return False
        return True

    def get_chain_data(self):
        """Get the blockchain data in a format suitable for display."""
        return [
            {
                'index': block.index,
                'timestamp': datetime.datetime.fromtimestamp(block.timestamp),
                'transactions': block.transactions,
                'previous_hash': block.previous_hash,
                'hash': block.hash,
                'nonce': block.nonce
            }
            for block in self.chain
        ]
