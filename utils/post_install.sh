    if [ -f ".venv/lib64/python3.10/site-packages/metaworld/assets/objects/assets/xyz_base.xml" ]; then
        if ! grep -q '<camera name="custom" fovy="45" mode="fixed" pos="0.65 -0.1 0.70" euler="4.0 2.6 0.40"/>' .venv/lib64/python3.10/site-packages/metaworld/assets/objects/assets/xyz_base.xml; then
            sed -i '20i\    <camera name="custom" fovy="45" mode="fixed" pos="0.65 -0.1 0.70" euler="4.0 2.6 0.40"/>' .venv/lib64/python3.10/site-packages/metaworld/assets/objects/assets/xyz_base.xml
        fi
    fi