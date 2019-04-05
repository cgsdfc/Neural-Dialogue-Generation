# Persona_Addressee

The persona_addressee model described in [2]

## Additional Options

    -train_file     (default ../data/speaker_addresseet_train.txt)
    -dev_file       (default ../data/speaker_addressee_dev.txt)
    -test_file      (default ../data/speaker_addressee_test.txt)
    -SpeakerNum     (default 10000, number of distinct speakers)
    -AddresseeNum   (default 10000, number of distinct addressees)
    -speakerSetting (taking values of "speaker" or "speaker_addressee". For "speaker", only the user who speaks is modeled. For "speaker_addressee" both the speaker and the addressee are modeled)

## Dataset Format
The first token of a source line is the index of the Addressee and the first token in the target line is the index of the speaker. For example: ``2 45 43 6|1 123 45`` means that the index of the addressee is 2 and the index of the speaker is 1

## Commands
To train the model, run

    th Persona/train.lua [params]

## Tests

    Persona/test_persona.sh
    Persona/test_persona_speaker.sh
    Persona/test_persona_speaker_addressee.sh