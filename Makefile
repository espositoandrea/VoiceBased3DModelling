.PHONY: dist

dist: VoiceBased3DModelling.zip

VoiceBased3DModelling.zip: __init__.py utilities.py requirements.txt
	cd .. && zip -r $(patsubst %.zip,%,$@)/$@ $(patsubst %.zip,%,$@) -x "*.git/*" -x "*.idea/*" -x $@ -x "*.venv/*"