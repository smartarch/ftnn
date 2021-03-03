package trust40.enforcer.sdq;

import trust40.k4case.AllowPermission;
import trust40.k4case.DenyPermission;

import java.util.List;
import java.util.Map;

/**
 *
 */
public interface DesignTimeDecisionMaker {

    /**
     * Checks the validity of the {@link AllowPermission} with the {@link DenyPermission} and the privacy levels
     *
     * @param allowPermissions  {@link List} of {@link AllowPermission} to check
     * @param denyPermissions {@link List} of {@link DenyPermission} to check
     * @param deniedPermissions {@link Map} with the denied {@link AllowPermission} and the reason
     * @return {@link List} with valid {@link AllowPermission}
     * @throws NullPointerException
     */
    List<AllowPermission> validatePolicies(List<AllowPermission> allowPermissions, List<DenyPermission> denyPermissions, Map<AllowPermission, List<DenyPermission>> deniedPermissions) throws NullPointerException;

}
