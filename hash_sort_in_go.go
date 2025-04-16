package main

import (
    "crypto/sha256"
    "fmt"
    "sort"
    "strconv"
)

func main() {
    strings := []string{"apple", "banana", "cherry", "date"}

    hashStringMap := make(map[string]string)
    for _, str := range strings {
        hash := sha256.Sum256([]byte(str))
        hashStringMap[str] = fmt.Sprintf("%x", hash)
    }

    var sortedHashes []string
    for _, hash := range hashStringMap {
        sortedHashes = append(sortedHashes, hash)
    }

    sort.Strings(sortedHashes)

    fmt.Println("Sorted hashes:")
    for idx, hash := range sortedHashes {
        fmt.Printf("%d: %s\n", idx+1, hash)
    }
}