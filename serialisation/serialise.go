package serialisation

import (
	"compress/gzip"
	"encoding/gob"
	"fmt"
	"io"
	"os"
)

func SerialiseTo(t interface{}, w io.Writer) error {
	gzWriter := gzip.NewWriter(w)
	encoder := gob.NewEncoder(gzWriter)
	err := encoder.Encode(t)
	if err != nil {
		return err
	}
	return gzWriter.Close()
}

func SerialiseToFile(t interface{}, filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("unable to open file: %s", err)
	}
	return SerialiseTo(t, f)
}

func DeserialiseFrom(t interface{}, r io.Reader) error {
	gzReader, err := gzip.NewReader(r)
	if err != nil {
		return err
	}

	decoder := gob.NewDecoder(gzReader)
	err = decoder.Decode(t)
	if err != nil {
		return err
	}

	return gzReader.Close()
}

func DeserialiseFromFile(t interface{}, filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("unable to open file: %s", err)
	}
	return DeserialiseFrom(t, f)
}
