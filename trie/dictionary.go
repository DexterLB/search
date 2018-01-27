package trie

type Dictionary struct {
	Trie Trie
	Size int32

	// if the dictionary is Closed, getting a word that's not already in the
	// dictionary will yield -1
	Closed bool
}

type BiDictionary struct {
	Dictionary

	Inverse map[int32][]byte
}

func NewDictionary() *Dictionary {
	return &Dictionary{
		Trie:   *New(),
		Size:   0,
		Closed: false,
	}
}

func (d *Dictionary) Get(word []byte) int32 {
	if d.Closed {
		idP := d.Trie.Get(word)
		if idP == nil {
			return -1
		}
		return *idP
	}

	id := d.Trie.GetOrPut(word, d.Size)
	if id == d.Size {
		d.Size += 1
	}

	return id
}

func NewBiDictionary() *BiDictionary {
	return &BiDictionary{
		Dictionary: *NewDictionary(),
		Inverse:    make(map[int32][]byte),
	}
}

func (b *BiDictionary) Get(word []byte) int32 {
	id := b.Dictionary.Get(word)
	if _, ok := b.Inverse[id]; !ok {
		b.Inverse[id] = append([]byte(nil), word...) // make a copy of word
	}
	return id
}

func (b *BiDictionary) GetInverse(id int32) []byte {
	word, ok := b.Inverse[id]
	if ok {
		return word
	} else {
		return nil
	}
}
