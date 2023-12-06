package main

import (
	"github.com/google/syzkaller/pkg/db"
	"github.com/google/syzkaller/pkg/log"
)

type Parser struct {
	corpusDB *db.DB
}

func LoadCorpus() {
	corpusDB, err := db.Open("./corpus.db", true)
	if err != nil {
		if corpusDB == nil {
			log.Fatalf("failed to open corpus database: %v", err)
		}
		log.Errorf("read %v inputs from corpus and got error: %v", len(corpusDB.Records), err)
	}

	for key, rec := range corpusDB.Records {
		nothing(key, rec)
	}
}

func nothing(k string, r db.Record) {}
