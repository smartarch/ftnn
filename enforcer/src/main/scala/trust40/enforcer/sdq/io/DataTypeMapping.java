package trust40.enforcer.sdq.io;

import java.io.IOException;
import java.util.Arrays;
import java.util.Map;
import java.util.stream.Collectors;

public class DataTypeMapping extends CSVLoader {
    public DataTypeMapping(String path) {
        super(path);
    }
    private Map<String, String> createMap(String[][] datatypeMap) {
        if(datatypeMap == null)
            throw new IllegalArgumentException("Privacy Map can't be null");
        if (datatypeMap[0].length != 2)
            throw new IllegalArgumentException("Size of 2d dimension needs to be 2");
        return Arrays.stream(datatypeMap).collect(Collectors.toMap(e->e[0],e->e[1]));
    }

    public Map<String, String> getTypeMapping() throws IOException {
        String[][] privacyStrings = super.loadCSVFile(2);
        return createMap(privacyStrings);
    }
}
