package trust40.enforcer.sdq.io;

import trust40.enforcer.sdq.data.PrivacyLevel;
import trust40.enforcer.sdq.data.PrivacyTable;
import java.io.IOException;

/**
 * Class for loading the privacy level
 */
public class PrivacyLoader extends CSVLoader {

    public PrivacyLoader(String path) {
        super(path);
    }

    /**
     * Loads the privacy level and returns a PrivacyTable with the triple (subject, operation, object) and the according privacy level
     *
     * @return {@link PrivacyTable} with the {@link PrivacyLevel}
     * @throws IOException If there are errors during the file reading
     */
    public PrivacyTable getPrivacyTable() throws IOException {
        String[][] privacyStrings = loadCSVFile(4);
        return new PrivacyTable(privacyStrings);
    }
}
