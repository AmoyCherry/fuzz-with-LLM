package main

import (
	"fmt"
	"github.com/google/syzkaller/pkg/db"
	"github.com/google/syzkaller/pkg/log"
	"os"
	"strconv"
)

type Parser struct {
	corpusDB *db.DB
}

const TokenFilePath = "./tokens/tokens_"
const DBCorpusPath = "./parser/old-corpus/corpus.db"
const TokenFileSize = 256
const VocabFilePath = "./vocab/vocab.txt"

func NewParser() *Parser {
	corpusDB, err := db.Open(DBCorpusPath, true)
	if err != nil {
		if corpusDB == nil {
			log.Fatalf("failed to open corpus database: %v", err)
		}
		log.Errorf("read %v inputs from corpus and got error: %v", len(corpusDB.Records), err)
	}

	parser := new(Parser)
	parser.corpusDB = corpusDB
	return parser
}

func (parser *Parser) Parse() {
	parser.CreatePathIfNotExist("./tokens/")
	parser.CreatePathIfNotExist("./vocab/")
	buffer := ""
	fileCount := 1
	sequenceCount := 0
	totalCount := 0
	for _, rec := range parser.corpusDB.Records {
		buffer += string(rec.Val[:]) + "[SEP]\n"
		sequenceCount += 1
		totalCount += 1

		if sequenceCount >= TokenFileSize {
			parser.WriteToFile(fileCount, &buffer)
			parser.WriteVocabFile(&buffer)

			fileCount += 1
			sequenceCount = 0
			buffer = ""
		}
	}
	buffer += "[UNK]\n"
	buffer += "[MASK]\n"
	buffer += "[CLS]\n"
	buffer += "[PAD]\n"
	parser.WriteToFile(fileCount, &buffer)
	parser.WriteVocabFile(&buffer)

	log.Logf(0, "total number of sequences (traces): %v", totalCount)
}

func (parser *Parser) CreatePathIfNotExist(path string) {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		err := os.MkdirAll(path, os.ModePerm)
		if err != nil {
			fmt.Println("Failed to create folder:", err)
			return
		}
		fmt.Println("Folder created successfully!")
	} else {
		fmt.Println("Folder already exists!")
	}
}

func (parser *Parser) WriteToFile(fileCount int, content *string) {
	f, err := os.Create(TokenFilePath + strconv.Itoa(fileCount) + ".txt")
	if err != nil {
		log.Fatalf("open token error :", err)
		return
	}

	_, err = f.Write([]byte(*content))
	if err != nil {
		log.Fatalf("write token error: ", err)
		return
	}

	err = f.Close()
	if err != nil {
		log.Fatalf("Close token error: ", err)
		return
	}
}

func (parser *Parser) WriteVocabFile(content *string) {
	f, err := os.OpenFile(VocabFilePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatalf("open token error :", err)
		return
	}

	_, err = f.Write([]byte(*content))
	if err != nil {
		log.Fatalf("write token error: ", err)
		return
	}

	err = f.Close()
	if err != nil {
		log.Fatalf("Close token error: ", err)
		return
	}
}
