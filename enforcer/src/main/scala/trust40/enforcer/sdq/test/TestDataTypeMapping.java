package trust40.enforcer.sdq.test;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import trust40.enforcer.sdq.data.DataObject;
import trust40.enforcer.sdq.data.Operation;
import trust40.enforcer.sdq.data.PrivacyLevel;
import trust40.enforcer.sdq.io.DataTypeMapping;
import trust40.enforcer.sdq.io.PrivacyLoader;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class TestDataTypeMapping {
    static Path t;

    @BeforeAll
    static void init() {
        t = Paths.get("test.csv");
        try (BufferedWriter writer = Files.newBufferedWriter(t, Charset.forName("UTF-8"))) {
            writer.write("A-foreman;foreman\n");
            writer.write("A-worker-001;worker\n");
            writer.write("A-worker-002;worker\n");
        } catch (IOException e) {
            e.printStackTrace();
            fail("IO-Error");
        }
    }
    @Test
    void testLoadMapping() {
        try {
            Map<String, String> mapping = new DataTypeMapping(t.toString()).getTypeMapping();
            assertAll("test Privacylevel",
                    ()-> assertEquals("foreman",mapping.get("A-foreman")),
                    ()-> assertEquals("worker",mapping.get("A-worker-001")),
                    ()-> assertEquals("worker",mapping.get("A-worker-002"))

            );
        } catch (IOException e) {
            e.printStackTrace();
            fail("IO-Error");
        }

    }
    @AfterAll
    static void clean() {
        try {
            Files.delete(t);
        } catch (IOException e) {
            e.printStackTrace();
            fail("IO-Error: can't delete mapping file");
        }
    }
}
